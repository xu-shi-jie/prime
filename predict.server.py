# fmt: off
import os
import sys
import io
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import urllib.request
import urllib.error
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.amp
import lightning as L
from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger
from biotite.structure.io import pdb, pdbx
from biotite.structure import Atom, to_sequence
import biotite.structure as struc
import biotite.structure.io as strucio
from scipy.cluster.hierarchy import fclusterdata
from tqdm import tqdm

from models.dataset import considered_metals, build_map, _1to3, elem2token
from models.module import Header
from models.plm import get_model, EsmModelInfo
from models.resnet import generate_model
from models.pdb import get_nos_atoms, rand_rot, num_sites, num_sites_ranges, get_probe
from models.utils import Config, safe_dist, backbone

torch.set_float32_matmul_precision("medium")


# Compiled functions
@torch.compile()
def cut_atoms(coords, center, dev, cutoff, rot=None, p='infinity'):
    """Cut atoms around the center coordinate"""
    vec = coords - center.to(dev)
    if rot is not None:
        vec = torch.matmul(vec, rot)
    if p == 'infinity':
        dist = vec.abs().max(dim=-1).values
    elif p == 2:
        dist = vec.norm(dim=-1)
    mask = dist < cutoff
    return vec[mask], mask


class ModelManager:
    """Manages all models in memory for fast prediction"""

    def __init__(self, plm_name="esm2_t33_650M_UR50D", stats_file='data/biolip/statistics_distance_residue.csv'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plm_name = plm_name
        self.stats_file = stats_file

        # Model caches
        self.seq_models = {}
        self.structure_models = {}
        self.plm_func = None
        self.processed_stats = {}
        self.stats_bounds = {}

        # Configs
        self.train_structure_args = Config('configs/train_structure_predictor.yaml')

        logger.info(f"Initializing ModelManager on device: {self.device}")

        if torch.cuda.is_available():
            avai_gpu_mem_GB = torch.cuda.mem_get_info()[0] / 1024 ** 3
            if avai_gpu_mem_GB > 60:
                self.batch_size = 128
            elif avai_gpu_mem_GB > 18:
                self.batch_size = 32
            else:
                self.batch_size = 1
            logger.info(f"{torch.cuda.get_device_name(0)}: {avai_gpu_mem_GB:.1f} GB available. Batch size: {self.batch_size}")
        else:
            logger.critical("CUDA not available. PRIME requires CUDA.")
            sys.exit(1)

        # Load PLM
        logger.info(f"Loading protein language model: {plm_name}")
        self.plm_func = get_model(plm_name, self.device)
        logger.success("PLM loaded successfully")

        # Load statistics
        self._load_statistics()

    def _load_statistics(self):
        """Load metal-residue distance statistics"""
        logger.info(f"Loading statistics from {self.stats_file}")
        stats = pd.read_csv(self.stats_file, keep_default_na=False)

        for metal in considered_metals:
            metal_stats = stats[stats['Metal'] == metal]
            if len(metal_stats) == 0:
                continue

            lb, ub = metal_stats['L'].min(), metal_stats['H'].max()
            self.stats_bounds[metal] = (lb, ub)

            processed = {}
            for i, row in metal_stats.iterrows():
                metal_name, res, atom, low, high, num = row['Metal'], row['Residue name'], row['Atom name'], row['L'], row['H'], row['Num']
                num /= metal_stats['Num'].sum()
                processed[f'{res},{atom},{metal_name}'] = [low, high, num]

            self.processed_stats[metal] = processed

        logger.success(f"Statistics loaded for {len(self.processed_stats)} metals")

    def load_models_for_metal(self, metal: str):
        """Load sequence and structure models for a specific metal"""
        if metal in self.seq_models and metal in self.structure_models:
            logger.info(f"Models for {metal} already loaded")
            return

        logger.info(f"Loading models for metal: {metal}")

        # Load sequence model
        seq_ckpt = f'checkpoints/seq_predictors/{metal}.ckpt'
        if not os.path.exists(seq_ckpt):
            raise FileNotFoundError(f"Sequence checkpoint not found: {seq_ckpt}")

        probe_args = Config(f"configs/seq_predictors/{metal}.yaml")
        state_dict = torch.load(seq_ckpt, self.device, weights_only=True)["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        esm_info = EsmModelInfo(self.plm_name)
        seq_model = Header(
            net_type=probe_args.rnn_type,
            in_size=esm_info['dim'],
            hidden_size=probe_args.rnn_hidden_size,
            num_layers=probe_args.rnn_layers,
            num_classes=2,
        ).to(self.device)
        seq_model.load_state_dict(state_dict)
        seq_model.eval()
        seq_model = torch.compile(seq_model)

        self.seq_models[metal] = (seq_model, probe_args)
        logger.success(f"Sequence model for {metal} loaded")

        # Load structure model
        ckpt_files = list(Path('weights').glob(f'probe_{metal}_*.ckpt'))
        if not ckpt_files:
            raise FileNotFoundError(f"No structure checkpoint found for {metal}")

        ckpt_file = str(ckpt_files[0])
        cnn_layers = int(re.findall(r'resnet(\d+)_', Path(ckpt_file).stem)[0])

        state_dict = torch.load(ckpt_file, self.device, weights_only=True)["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        structure_model = generate_model(cnn_layers, n_classes=5, num_bins=0.3, grid_dim=60).to(self.device)
        structure_model.load_state_dict(state_dict)
        structure_model.eval()
        structure_model = torch.compile(structure_model)

        self.structure_models[metal] = structure_model
        logger.success(f"Structure model for {metal} loaded")

    def load_all_models(self):
        """Preload all models for all metals"""
        logger.info("Preloading all models...")
        for metal in considered_metals:
            try:
                self.load_models_for_metal(metal)
            except Exception as e:
                logger.warning(f"Failed to load models for {metal}: {e}")
        logger.success("All models loaded successfully!")

    def predict(self, pdb_file: str, metal: str, max_rot: int = 5, cluster_dist: float = 3.0,
                probe_thresh: float = 0.5, cutoff: float = 10.0, max_sites: int = 10000) -> List[Dict]:
        """
        Run prediction using pre-loaded models

        Returns:
            List of predicted sites with position, confidence, residue_id, element
        """
        # Ensure models are loaded
        if metal not in self.seq_models or metal not in self.structure_models:
            self.load_models_for_metal(metal)

        seq_model, probe_args = self.seq_models[metal]
        structure_model = self.structure_models[metal]
        lb, ub = self.stats_bounds[metal]
        processed_stats = self.processed_stats[metal]

        # Load structure
        logger.info(f"Processing {pdb_file} for {metal}")
        if pdb_file.endswith(".pdb"):
            atoms = pdb.get_structure(pdb.PDBFile.read(pdb_file))[0]
        elif pdb_file.endswith(".cif"):
            atoms = pdbx.get_structure(pdbx.CIFFile.read(pdb_file))[0]
        else:
            raise ValueError("Unsupported file type. Use PDB or CIF.")

        # Extract chains
        chains = {}
        aa_atoms = atoms[(atoms.hetero == False) & np.isin(atoms.res_name, list(_1to3.values()))]

        for seq, chain_id_idx in zip(*to_sequence(aa_atoms)):
            seq = str(seq)
            chain_id = aa_atoms.chain_id[chain_id_idx]
            if chain_id.startswith('sym'):
                continue
            emb = self.plm_func([seq])[..., 1].half()
            chains[chain_id] = (seq, emb)

        logger.info(f"{len(chains)} chain(s) found, max length: {max([len(seq) for seq, _ in chains.values()])}")

        if len(chains) == 0:
            return []

        # Sequence prediction
        candidates = {}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
            num_sites_range = num_sites_ranges.get(metal, (100, 10000))
            for chain_id, (seq, emb) in chains.items():
                logits = seq_model(emb.to(self.device).unsqueeze(0).float()).detach()
                prob = torch.softmax(logits[0], dim=-1)[..., 1]
                pred = prob > probe_args.detect_thresh
                min_sites = num_sites(seq)

                if pred.sum() < max(min_sites, num_sites_range[0]):
                    pred = torch.zeros_like(pred, dtype=torch.bool)
                    pred[prob.argsort(descending=True)[:min_sites]] = True
                elif pred.sum() > min(num_sites_range[1], max_sites):
                    pred[prob.argsort(descending=True)[:max_sites]] = True

                candidates[chain_id] = (seq, pred, prob)

        logger.info(f"{sum([p[1].sum() for p in candidates.values()])} residue(s) detected")

        # Generate probes
        NOS_coords, NOS_elem, NOS_resnia, NOS_probs = [], [], [], []
        for chain_id, (seq, pred, prob) in candidates.items():
            backbone_atoms = backbone(aa_atoms, chain_id)
            if len(backbone_atoms) == 0:
                continue
            seq2res = build_map(seq, backbone_atoms)

            for i in torch.nonzero(pred).flatten():
                nos = get_nos_atoms((i, prob.cpu().numpy(), seq2res, atoms, chain_id, metal,
                                    processed_stats, self.train_structure_args.probe_res_thresh))
                if nos is not None:
                    NOS_coords.append(torch.tensor(nos[0]).view(-1, 3))
                    NOS_elem.extend(nos[1])
                    NOS_probs.extend(nos[2])
                    NOS_resnia.extend(nos[3])

        if len(NOS_coords) == 0:
            logger.warning("No NOS atoms found")
            return []

        ALL_coords = torch.tensor(aa_atoms.coord).to(self.device)
        tmp_nos_coords = torch.cat(NOS_coords, dim=0).to(self.device)
        mask = safe_dist(tmp_nos_coords, ALL_coords) < probe_args.max_offset
        ALL_coords = ALL_coords[mask]

        # Get probes
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
            probes = get_probe(
                metal, processed_stats, NOS_coords, NOS_probs, NOS_resnia, NOS_elem,
                ALL_coords, 0.5, self.device, lb, ub,
                self.train_structure_args.probe_score_thresh,
                probe_args.detect_thresh, probe_args.max_offset,
            ).float()
            probes, kskps = torch.split(probes, [3, 2], dim=-1)

        logger.info(f'{len(probes)} probe(s) generated')

        # Structure prediction
        clean_atoms = atoms[np.isin(atoms.element, list(elem2token.keys()))]
        elements = torch.tensor([elem2token[e] for e in clean_atoms.element]).to(self.device)
        coords = torch.tensor(clean_atoms.coord).to(self.device)

        predictions = []
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for rot_idx in range(max_rot):
                rot = rand_rot(dev=self.device) if max_rot > 1 else None

                for i in range(0, len(probes), self.batch_size):
                    data = []
                    for j, probe in enumerate(probes[i:i+self.batch_size]):
                        r_coords, mask = cut_atoms(coords, probe, self.device, cutoff, rot=rot)
                        data.append((
                            j * torch.ones(len(r_coords), dtype=torch.long).to(self.device),
                            elements[mask].to(self.device),
                            r_coords,
                        ))

                    bi, elems, r_coords = zip(*data)
                    x = torch.cat([
                        torch.cat(bi).unsqueeze(-1),
                        torch.cat(elems).unsqueeze(-1),
                        torch.cat(r_coords),
                    ], dim=-1).to(self.device)

                    probe = probes[i:i+self.batch_size]
                    y_pred = structure_model(x.to(torch.bfloat16))

                    mask = y_pred[..., 2:5].norm(dim=-1) > probe_args.max_offset
                    y_pred[..., :2][mask] = torch.tensor([1, 0], dtype=torch.bfloat16).to(self.device)

                    if rot is not None:
                        y_pred[..., 2:5] = y_pred[..., 2:5] @ rot.T + probe.to(self.device)

                    predictions.extend(y_pred)

        if len(predictions) == 0:
            logger.warning("No binding sites detected")
            return []

        # Post-processing
        predictions = torch.stack(predictions).to(torch.float32).reshape(-1, max_rot, 5)
        y_pred_logits, y_pred_offsets = predictions.split([2, 3], dim=-1)

        y_probs = torch.softmax(y_pred_logits, dim=-1)[..., 1].max(dim=-1)
        y_pred_offsets = y_pred_offsets[torch.arange(y_pred_offsets.shape[0]), y_probs.indices].reshape(-1, 3)
        y_probs = y_probs.values

        if len(y_pred_offsets) == 1:
            groups = np.array([0])
        else:
            groups = fclusterdata(y_pred_offsets.cpu().numpy(), cluster_dist, criterion='distance')

        centers = np.unique(groups)

        results = []
        for center_idx in centers:
            cluster = y_pred_offsets[groups == center_idx]
            prob = y_probs[groups == center_idx]

            if len(cluster) <= 1:
                continue

            center = cluster[prob.argmax()]
            bf = prob.max().cpu().numpy().item()

            if bf < probe_thresh:
                continue

            d = (center - coords).norm(dim=-1).min()
            if d < lb or d > ub:
                continue

            results.append({
                'position': center.cpu().numpy().tolist(),
                'confidence': float(bf),
                'residue_id': int(1000 + len(results)),
                'element': metal
            })

            logger.info(f"Site {len(results)}: prob={bf:.3f}, pos={center.cpu().numpy()}")

        logger.success(f"{len(results)} binding site(s) found")
        return results


# Initialize Flask app
app = Flask(__name__)
CORS(app)
lock = threading.Lock()
model_manager = None


def download_structure(protein_id: str, output_path: str) -> bool:
    """Download structure from PDB or Uniprot ID"""
    try:
        protein_id = protein_id.strip().upper()

        if len(protein_id) == 4:
            logger.info(f"Downloading PDB: {protein_id}")
            pdb_url = f"https://files.rcsb.org/download/{protein_id}.pdb"
            try:
                urllib.request.urlretrieve(pdb_url, output_path)
                logger.success(f"Downloaded: {protein_id}.pdb")
                return True
            except urllib.error.HTTPError:
                cif_url = f"https://files.rcsb.org/download/{protein_id}.cif"
                try:
                    urllib.request.urlretrieve(cif_url, output_path)
                    logger.success(f"Downloaded: {protein_id}.cif")
                    return True
                except:
                    pass

        # Try AlphaFold
        logger.info(f"Downloading AlphaFold structure: {protein_id}")
        alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-2-F1-model_v6.cif"
        urllib.request.urlretrieve(alphafold_url, output_path)
        logger.success(f"Downloaded AlphaFold structure")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


@app.route('/predict/', methods=['POST'])
def predict():
    """API endpoint for metal binding prediction"""
    # Check if server is busy
    lock_acquired = lock.acquire(blocking=False)
    if not lock_acquired:
        logger.warning("Request rejected: server busy")
        return jsonify({
            "message": "<span style='color: red'>Server busy. Please try again.</span>"
        }), 429

    logger.info("Lock acquired, processing prediction request")
    temp_input_file = None
    pdb_id = None

    try:
        metal_type = request.form.get('metal_type', 'ALL').upper()
        protein_id = request.form.get('protein_id', '').strip()

        # Handle file upload or protein ID
        if 'structure_file' in request.files and request.files['structure_file'].filename:
            uploaded_file = request.files['structure_file']
            filename = uploaded_file.filename.lower()

            ext = '.pdb' if filename.endswith(('.pdb', '.ent')) else '.cif' if filename.endswith('.cif') else None
            if ext is None:
                return jsonify({
                    "message": "<span style='color: red'>Invalid file format. Use PDB or CIF.</span>"
                }), 400

            temp_input_file = tempfile.NamedTemporaryFile(mode='wb', suffix=ext, delete=False).name
            uploaded_file.save(temp_input_file)
            logger.info(f"File uploaded: {temp_input_file}")

        elif protein_id:
            pdb_id = protein_id.upper()
            temp_input_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pdb', delete=False).name

            if not download_structure(protein_id, temp_input_file):
                return jsonify({
                    "message": f"<span style='color: red'>Failed to download: {protein_id}</span>"
                }), 400
        else:
            return jsonify({
                "message": "<span style='color: red'>Please upload a file or provide PDB/Uniprot ID.</span>"
            }), 400

        # Run predictions
        metals_to_predict = considered_metals if metal_type == 'ALL' else [metal_type]

        if metal_type != 'ALL' and metal_type not in considered_metals:
            return jsonify({
                "message": f"<span style='color: red'>Invalid metal: {metal_type}</span>"
            }), 400

        all_binding_sites = []
        results_by_metal = {}

        for metal in metals_to_predict:
            try:
                predictions = model_manager.predict(temp_input_file, metal)
                if predictions:
                    results_by_metal[metal] = predictions
                    all_binding_sites.extend(predictions)
            except Exception as e:
                logger.error(f"Prediction failed for {metal}: {e}")

        # Generate HTML message
        if not all_binding_sites:
            message = "<div class='alert alert-warning'>No metal-binding sites detected.</div>"
        else:
            message = "<div class='alert alert-success'><strong>Prediction Complete!</strong></div>"
            message += f"<p>Found <strong>{len(all_binding_sites)}</strong> potential metal-binding site(s)</p>"

            message += "<h5>Results by Metal Type:</h5>"
            message += "<table class='table table-striped table-sm'>"
            message += "<thead><tr><th>Metal</th><th>Sites</th><th>Top Confidence</th></tr></thead><tbody>"

            for metal, predictions in sorted(results_by_metal.items()):
                num_sites = len(predictions)
                top_conf = max([p['confidence'] for p in predictions])
                message += f"<tr><td><strong>{metal}</strong></td><td>{num_sites}</td><td>{top_conf:.3f}</td></tr>"

            message += "</tbody></table>"

            message += "<h5>Top Predicted Sites:</h5>"
            message += "<table class='table table-sm'>"
            message += "<thead><tr><th>Residue</th><th>Confidence</th><th>Element</th></tr></thead><tbody>"

            sorted_sites = sorted(all_binding_sites, key=lambda x: x['confidence'], reverse=True)[:10]
            for site in sorted_sites:
                message += f"<tr><td>Residue {site['residue_id']}</td>"
                message += f"<td>{site['confidence']:.3f}</td>"
                message += f"<td>{site['element']}</td></tr>"

            message += "</tbody></table>"

        # Generate PDB file with predicted metal ions
        predicted_pdb_content = None
        if all_binding_sites:
            # Create biotite Atom array with predicted metal ions
            predicted_atoms = []
            for i, site in enumerate(all_binding_sites):
                atom = Atom(
                    element=site['element'],
                    res_name=site['element'],
                    res_id=site['residue_id'],
                    chain_id='Z',  # Use chain Z for predicted metals
                    coord=np.array(site['position']),
                    hetero=True,
                )
                predicted_atoms.append(atom)

            # Convert to biotite array
            atom_array = struc.array(predicted_atoms)
            atom_array.add_annotation('b_factor', float)
            atom_array.b_factor = np.array([site['confidence'] for site in all_binding_sites])

            # Save to string (in-memory)
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, atom_array)
            output = io.StringIO()
            pdb_file.write(output)
            predicted_pdb_content = output.getvalue()
            logger.info(f"Generated PDB with {len(predicted_atoms)} metal ions")

        # Prepare response
        response = {"message": message, "binding_sites": all_binding_sites}

        if pdb_id:
            response["pdb_id"] = pdb_id

        # Add predicted structure (metal ions only)
        if predicted_pdb_content:
            response["predicted_structure"] = predicted_pdb_content

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "message": f"<span style='color: red'>Error: {str(e)}</span>"
        }), 500

    finally:
        # Always release lock if it was acquired
        try:
            lock.release()
            logger.info("Lock released successfully")
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")

        # Clean up temporary file
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.unlink(temp_input_file)
                logger.debug(f"Temp file deleted: {temp_input_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "available_metals": considered_metals,
        "models_loaded": list(model_manager.seq_models.keys()) if model_manager else []
    }), 200


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="<fg #03832e>🌳 PRIME Server</fg #03832e> <level>[{level}] {message}</level>")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind to")
    parser.add_argument("--preload", action="store_true", help="Preload all models on startup")
    args = parser.parse_args()

    # Initialize model manager
    logger.info("Initializing PRIME Server...")
    model_manager = ModelManager()

    if args.preload:
        model_manager.load_all_models()
    else:
        logger.info("Models will be loaded on-demand (use --preload to load all at startup)")

    logger.success(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Available metals: {', '.join(considered_metals)}")

    # Use threaded=True to handle multiple requests, but lock ensures only one prediction at a time
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
