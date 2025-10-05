import warnings
import argparse
import shutil
import line_profiler
import rich.rule
import torch
import torch.amp
from tqdm import tqdm
import re
import lightning as L
import torch.nn as nn
from models.module import Header
from models.plm import get_model, EsmModelInfo
from models.dataset import build_map, _1to3, elem2token
from models.resnet import generate_model
from models.pdb import generate_pymol_script, get_nos_atoms, rand_rot, num_sites
from biotite.structure.io import pdb, pdbx
from Bio.SeqIO import parse
from loguru import logger
from math import ceil
import numpy as np
import pandas as pd
from biotite.structure import Atom, to_sequence, get_chains
import biotite.structure as struc
import biotite.structure.io as strucio
from scipy.cluster.hierarchy import fclusterdata
import time
import sys
import rich
from rich.console import Console
from pathlib import Path
from models.pdb import num_sites_ranges, get_probe
from models.training import ProbeModel
from models.utils import Config, safe_dist, segment_cmd, backbone, shorten_path
torch.set_float32_matmul_precision("medium")


# @line_profiler.profile
@torch.compile()
def cut_atoms(coords, center, dev, cutoff, rot=None, p='infinity',):
    """ Cut atoms around the center coordinate
    Args:
        coords: torch.tensor, shape=(N, 3), The coordinates of atoms
        center: torch.tensor, shape=(3,), The center coordinate
        dev: torch.device, The device
        cutoff: float, The cutoff distance
        rot: torch.tensor, shape=(3, 3), The rotation matrix
        p: int, the norm of distance
    Returns:
        np.array, The cut atoms
    """
    vec = coords - center.to(dev)  # center the coordinates

    if rot is not None:  # rotate around the center
        vec = torch.matmul(vec, rot)

    if p == 'infinity':
        dist = vec.abs().max(dim=-1).values
    elif p == 2:
        dist = vec.norm(dim=-1)

    mask = dist < cutoff
    return vec[mask], mask


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', nargs='+', type=str, help='List of PDB files', default=['data/cifs/5xwk.cif'])
    parser.add_argument("--outdir", type=str, help="Output directory", default='.')
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file")
    parser.add_argument("--metal", type=str, default=None, help="Metal ion type")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file to structure model", default='checkpoints/probe_predictors/probe_ZN_resnet50_pretrained=True_hard_mining_epoch=52_val_f1=0.9079.ckpt')
    parser.add_argument("--cnn_layers", type=int, help="Number of CNN layers", default=None)

    # default hyperparameters
    parser.add_argument("--cluster_dist", type=float, default=3.0, help="Distance threshold for clustering")
    parser.add_argument("--max_rot", type=int, default=5, help="Rotation times for augmentation")
    parser.add_argument("--plm", type=str, default="esm2_t33_650M_UR50D", help="Protein language model name")
    parser.add_argument("--stats", type=str, default='data/biolip/statistics_distance_residue.csv', help="Statistics file of metal-NOS-AA distances")
    parser.add_argument("--probe_size", type=float, default=0.5, help="Probe size")
    parser.add_argument("--max_sites", type=int, default=10000, help="Maximum number of sites")
    parser.add_argument("--cutoff", type=float, default=10.0, help="Cutoff distance for cutting atoms")
    parser.add_argument("--probe_thresh", type=float, default=0.5, help="Threshold for probe detection")
    parser.add_argument("--detect_thresh", type=float, help="Threshold for sequence detection")
    parser.add_argument("--max_offset", type=float, default=None, help="Maximum offset for probe detection")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for probe processing, decrease if OOM")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timing", action="store_true", help="Timing the inference")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--output_probes", action="store_true", help="Output probes in PyMOL PSE format")
    args = parser.parse_args()
    # fmt: on

    logger.remove()
    logger.add(
        sys.stdout, format="<fg #03832e>🌳 PRIME</fg #03832e> <level>[{level}] {message}</level>")

    # logger.info(f'PRIME ver 1.0, developed by Shijie Xu, 2025. If you use this tool, please cite paper: <>. Type "python predict.py --help" for help.')

    if not args.verbose:
        logger.configure(handlers=[{"sink": sys.stdout, "level": "CRITICAL"}])

    if args.metal is None:
        args.metal = Path(args.ckpt).stem.split('_')[1]
    if args.cnn_layers is None:
        args.cnn_layers = int(re.findall(
            r'resnet(\d+)_', Path(args.ckpt).stem)[0])

    probe_args = Config(f"configs/seq_predictors/{args.metal}.yaml")
    train_structure_args = Config('configs/train_structure_predictor.yaml')
    if args.detect_thresh is not None:
        probe_args.detect_thresh = args.detect_thresh
    if args.max_offset is not None:
        probe_args.max_offset = args.max_offset

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        avai_gpu_mem_GB = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if avai_gpu_mem_GB > 60:  # we use H100 80GB
            args.batch_size = 128
        elif avai_gpu_mem_GB > 18:  # we use NVIDIA 4090 24GB
            args.batch_size = 32
        else:
            args.batch_size = 1
        logger.info(
            f"{torch.cuda.get_device_name(0)}: available memory {avai_gpu_mem_GB:.1f} GB. Use batch size {args.batch_size}.")
    else:
        logger.critical(
            "CUDA is not available. Current implementation requires CUDA.")
        exit(1)

    stats = pd.read_csv(args.stats, keep_default_na=False)
    stats = stats[stats['Metal'] == args.metal]
    lb, ub = stats['L'].min(), stats['H'].max()
    logger.info(
        f'Metal: {args.metal}, L: {lb:.2f}, H: {ub:.2f}, max_offset: {probe_args.max_offset:.2f}')
    processed_stats = {}
    for i, row in stats.iterrows():
        metal, res, atom, low, high, num = row['Metal'], row[
            'Residue name'], row['Atom name'], row['L'], row['H'], row['Num']
        num /= stats['Num'].sum()
        processed_stats[f'{res},{atom},{metal}'] = [low, high, num]

    L.seed_everything(args.seed, verbose=False)
    logger.info(f'Set random seed to {args.seed}.')

    state_dict = torch.load(
        f'checkpoints/seq_predictors/{args.metal}.ckpt', dev, weights_only=True)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    esm_info = EsmModelInfo(args.plm)
    seq_model = Header(
        net_type=probe_args.rnn_type,
        in_size=esm_info['dim'],
        hidden_size=probe_args.rnn_hidden_size,
        num_layers=probe_args.rnn_layers,
        num_classes=2,).to(dev)
    seq_model.load_state_dict(state_dict)
    seq_model.eval()
    if not args.no_compile:
        seq_model = torch.compile(seq_model)
    plm_func = get_model(args.plm, dev)
    logger.info("Sequence model loaded.")

    state_dict = torch.load(args.ckpt, dev, weights_only=True,)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    probe_model = generate_model(
        args.cnn_layers, n_classes=5,
        num_bins=0.3, grid_dim=60).to(dev)
    # probe_model = ProbeModel(args.cnn_layers).to(dev)
    probe_model.load_state_dict(state_dict)
    probe_model.eval()
    if not args.no_compile:
        probe_model = torch.compile(probe_model)
    logger.info("Structure model loaded.")

    total_start = time.time()
    total = len(list(args.pdb))
    for pdb_idx, pdb_file in enumerate(args.pdb):
        pdb_out = Path(
            args.outdir, f"{Path(pdb_file).stem}-prime-{args.metal}.pdb")

        if pdb_out.exists() and not args.overwrite:
            logger.info(
                f"Output file {shorten_path(str(pdb_out))} already exists. Skip.")
            continue

        if ':' in pdb_file:
            pdb_file, selected_chain = pdb_file.split(':')
        else:
            selected_chain = None

        logger.info(
            f"Processing {pdb_file} ......")

        start = time.time()
        # extract protein sequence from PDB file
        if pdb_file.endswith(".pdb"):
            atoms = pdb.get_structure(pdb.PDBFile.read(pdb_file))[0]
        elif pdb_file.endswith(".cif"):
            atoms = pdbx.get_structure(pdbx.CIFFile.read(pdb_file))[0]
        else:
            raise NotImplementedError("Unknown file type")

        chains = {}
        aa_atoms = atoms[
            (atoms.hetero == False) &  # exclude hetero atoms
            np.isin(atoms.res_name, list(_1to3.values()))]  # only include amino acids

        for seq, chain_id_idx in zip(*to_sequence(aa_atoms)):
            seq = str(seq)
            chain_id = aa_atoms.chain_id[chain_id_idx]
            if chain_id.startswith('sym'):
                continue
            # NOTE: in biotite, chaid_id such as B1 is treated as B
            if selected_chain is not None and chain_id != selected_chain[:len(chain_id)]:
                continue
            emb = plm_func([seq])[..., 1].half()
            chains[chain_id] = (seq, emb)
        logger.info(
            f"{len(chains)} chain(s) found: {selected_chain}.")
        if len(chains):
            logger.info(
                f"Max chain length: {max([len(seq) for seq, _ in chains.values()])}.")
        else:
            continue

        candidates = {}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
            num_sites_range = num_sites_ranges.get(args.metal, (100, 10000))
            for chain_id, (seq, emb) in chains.items():
                logits = seq_model(emb.to(dev).unsqueeze(0).float()).detach()
                prob = torch.softmax(logits[0], dim=-1)[..., 1]
                pred = prob > probe_args.detect_thresh
                min_sites = num_sites(seq)
                if pred.sum() < max(min_sites, num_sites_range[0]):
                    logger.info(
                        f"No enough seq_prob > {probe_args.detect_thresh} on chain {chain_id}, using top {min_sites}.")
                    pred = torch.zeros_like(pred, dtype=torch.bool)
                    pred[prob.argsort(descending=True)[:min_sites]] = True
                elif pred.sum() > min(num_sites_range[1], args.max_sites):
                    pred[prob.argsort(descending=True)[:args.max_sites]] = True
                candidates[chain_id] = (seq, pred, prob)

        logger.info(
            f"{sum([p[1].sum() for p in candidates.values()])} residue(s) detected.")

        NOS_coords, NOS_elem, NOS_resnia, NOS_probs = [], [], [], []
        for chain_id, (seq, pred, prob) in candidates.items():
            backbone_atoms = backbone(aa_atoms, chain_id)
            assert len(seq), f"Empty sequence: {seq}"
            if len(backbone_atoms) == 0:
                logger.warning(
                    f"Empty backbone atoms: {chain_id} of {args.pdb}")
                continue
            seq2res = build_map(seq, backbone_atoms)
            logger.opt(raw=True).info(
                f'\rProcessing chain {chain_id} ({len(seq)} AA) ......')

            for i in torch.nonzero(pred).flatten():
                nos = get_nos_atoms((i, prob.cpu().numpy(
                ), seq2res, atoms, chain_id, metal, processed_stats, train_structure_args.probe_res_thresh))
                if nos is not None:
                    NOS_coords.append(torch.tensor(nos[0]).view(-1, 3))
                    NOS_elem.extend(nos[1])
                    NOS_probs.extend(nos[2])
                    NOS_resnia.extend(nos[3])

        ALL_coords = torch.tensor(aa_atoms.coord).to(dev)
        tmp_nos_coords = torch.cat(NOS_coords, dim=0).to(dev)
        mask = safe_dist(
            tmp_nos_coords, ALL_coords) < probe_args.max_offset
        ALL_coords = ALL_coords[mask]

        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
                probes = get_probe(
                    args.metal,
                    processed_stats, NOS_coords, NOS_probs, NOS_resnia,
                    NOS_elem, ALL_coords, args.probe_size, dev,
                    lb, ub,
                    train_structure_args.probe_score_thresh,
                    probe_args.detect_thresh, probe_args.max_offset,
                ).float()
                probes, kskps = torch.split(probes, [3, 2], dim=-1)
        except Exception as e:
            continue
        # write probes
        if args.output_probes:
            import pymol
            pymol.cmd.reinitialize()
            pymol.finish_launching(['pymol', '-qc'])
            pymol.cmd.feedback("disable", "all", "everything")
            pymol.cmd.load(pdb_file)
            for c in segment_cmd(generate_pymol_script(probes)):
                pymol.cmd.do(c)
            pymol.cmd.save(f"{Path(pdb_file).stem}-prime-{args.metal}.pse")
            logger.info(
                f'Output probes saved to {Path(pdb_file).stem}-prime-{args.metal}.pse')
        logger.info(
            f'{len(probes)} probe(s) generated.')
        seq_infer_time = time.time() - start

        start = time.time()
        clean_atoms = atoms[np.isin(atoms.element, list(elem2token.keys()))]
        elements = torch.tensor([elem2token[e]
                                for e in clean_atoms.element]).to(dev)
        coords = torch.tensor(clean_atoms.coord).to(dev)
        predictions = []
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for rot_idx in tqdm(range(args.max_rot), desc=f"Rotating probes", leave=False):
                if args.max_rot > 1:
                    rot = rand_rot(dev=dev)
                else:
                    rot = None

                for i in tqdm(range(0, len(probes), args.batch_size), desc=f"Processing probes", leave=False):
                    data = []
                    for j, probe in enumerate(probes[i:i+args.batch_size]):
                        r_coords, mask = cut_atoms(
                            coords, probe, dev, args.cutoff, rot=rot)
                        data.append((
                            j*torch.ones(
                                len(r_coords), dtype=torch.long).to(dev),
                            elements[mask].to(dev),
                            r_coords,))
                    bi, elems, r_coords = zip(*data)
                    x = torch.cat([
                        torch.cat(bi).unsqueeze(-1),
                        torch.cat(elems).unsqueeze(-1),
                        torch.cat(r_coords),], dim=-1).to(dev)
                    probe = probes[i:i+args.batch_size]
                    kskp = kskps[i:i+args.batch_size]
                    # y_pred = probe_model(
                    #     x.to(torch.bfloat16), kskp.to(torch.bfloat16))
                    y_pred = probe_model(x.to(torch.bfloat16))

                    # check y_pred norm, and rotate back
                    mask = y_pred[..., 2:5].norm(
                        dim=-1) > probe_args.max_offset
                    y_pred[..., :2][mask] = torch.tensor(
                        [1, 0], dtype=torch.bfloat16).to(dev)
                    if rot is not None:
                        y_pred[..., 2:5]  \
                            = y_pred[..., 2:5] @ rot.T + probe.to(dev)
                    predictions.extend(y_pred)

        if len(predictions) == 0:
            logger.critical("No binding sites detected.")
            continue

        predictions = torch.stack(predictions).to(
            torch.float32).reshape(-1, args.max_rot, 5)
        y_pred_logits, y_pred_offsets = predictions.split([2, 3], dim=-1)

        y_probs = torch.softmax(y_pred_logits, dim=-1)[..., 1].max(dim=-1)
        y_pred_offsets = y_pred_offsets[torch.arange(
            y_pred_offsets.shape[0]), y_probs.indices].reshape(-1, 3)
        y_probs = y_probs.values

        cluster_dist = args.cluster_dist
        if len(y_pred_offsets) == 1:
            groups = np.array([0])
        else:
            groups = fclusterdata(
                y_pred_offsets.cpu().numpy(), cluster_dist, criterion='distance',
                # method='complete'
            )
        centers = np.unique(groups)

        predictions = []
        bfactors = []  # probability will be saved in b_factor
        sizes = []
        for center_idx in centers:
            cluster = y_pred_offsets[groups == center_idx]
            prob = y_probs[groups == center_idx]

            if len(cluster) <= 1:
                continue

            # choose the one with the highest probability
            center = cluster[prob.argmax()]
            bf = prob.max().cpu().numpy().item()

            if bf < args.probe_thresh:
                continue

            # check if the center is not close to other atoms
            d = (center-coords).norm(dim=-1).min()
            if d < lb or d > ub:
                continue

            ks, kp = torch.max(
                kskps[groups == center_idx],
                dim=0).values.split([1, 1], dim=-1)

            predictions.append(center)
            bfactors.append(bf)
            sizes.append(len(cluster))

            logger.info(
                f"Cluster {center_idx}:\t{len(cluster)} probe(s),\t"
                f"probe_prob: {bf:.2f},\t"
                f"ks: {ks.cpu().numpy().item():.2f},\tkp: {kp.cpu().numpy().item():.2f},\t"
                f"pos: [{', '.join([f'{x:.2f}' for x in center])}]")

        if len(predictions) == 0:
            logger.critical(
                f"No binding sites detected [{pdb_idx+1}/{len(args.pdb)}].")
            continue

        probe_infer_time = time.time() - start
        logger.info(
            f"{len(predictions)} site{'s' if len(predictions) > 1 else ''} found in {seq_infer_time+probe_infer_time:.2f}"
            f" (seq: {seq_infer_time:.2f} + probe: {probe_infer_time:.2f}) seconds.")

        atoms = []
        for i, center in enumerate(predictions):
            atoms.append(Atom(
                element=args.metal,
                res_name=args.metal,
                res_id=1000+i,  # start from 1000 to avoid conflict
                chain_id='Z',  # use Z to avoid conflict
                coord=center.cpu().numpy(),
                hetero=True,
            ))

        if len(atoms) == 0:
            logger.critical("No output file.")
            continue

        atoms = struc.array(atoms)
        atoms.add_annotation('b_factor', float)
        atoms.b_factor = np.array(bfactors)
        timing_out = pdb_out.with_suffix('.timing.csv')

        strucio.save_structure(pdb_out, atoms)
        logger.info(f"Output file saved to {pdb_out}.")
        if args.timing:
            df = pd.DataFrame(
                {'seq_infer_time': [seq_infer_time],
                 'probe_infer_time': [probe_infer_time],
                 'total_time': [seq_infer_time + probe_infer_time]})
            df.to_csv(timing_out, index=False)
            logger.info(f"Timing info saved to {timing_out}.")

        time_escaped = time.time() - total_start
        speed = time_escaped / (pdb_idx+1)
        estimated = (total - pdb_idx - 1) * speed
        logger.success(
            f'Finished processing [{pdb_idx+1}/{len(args.pdb)}], Escaped: {time_escaped:.2f}s, Remaining: {estimated:.2f}s')


if __name__ == "__main__":
    main()
