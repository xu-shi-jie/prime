import multiprocessing as mp
from pathlib import Path
import argparse
import shutil
import line_profiler
import os
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import rich
from biotite.structure.io.pdbx import CIFFile, get_structure
from math import ceil
import rich
from loguru import logger
# fmt: off
import sys
sys.path.append(".")
from models.module import Header, MLP
from models.utils import Config, safe_dist, segment_cmd,  backbone
from models.plm import EsmModelInfo
from models.pdb import num_sites, generate_pymol_script, get_probe, get_nos_atoms
from models.dataset import build_map, _1to3
# fmt: on
torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    os.system('rm -f *.pse *.cif *.pdb')

    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", type=str, default='CA')
    args = parser.parse_args()

    metal = args.metal

    conf_sp = Config(f"configs/seq_predictors/{metal}.yaml")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # spawn init
    torch.multiprocessing.set_start_method('spawn')

    shutil.rmtree(f'cache/probe/{metal}', ignore_errors=True)
    shutil.rmtree(f'probes/{metal}', ignore_errors=True)
    cache_dir = Path(f"cache", "probe", metal)
    cache_dir.mkdir(exist_ok=True, parents=True)
    Path('probes', metal).mkdir(exist_ok=True, parents=True)

    stats = pd.read_csv(
        'data/biolip/statistics_distance_residue.csv', keep_default_na=False)
    stats = stats[stats['Metal'] == metal]

    lb, ub = stats['L'].min(), stats['H'].max()
    logger.info(
        f"Metal: {metal}, L: {lb:.2f}, H: {ub:.2f}, max_offset: {conf_sp.max_offset:.2f}")
    processed_stats = {}
    for i, row in stats.iterrows():
        metal, res, atom, low, high, num = row['Metal'], row[
            'Residue name'], row['Atom name'], row['L'], row['H'], row['Num']
        num /= stats['Num'].sum()
        processed_stats[f'{res},{atom},{metal}'] = [low, high, num]

    state_dict = torch.load(conf_sp.pretrained_rnn)["state_dict"]
    biolip_state = {k.replace("model.", ""): v for k,
                    v in state_dict.items() if k.startswith("model.")}

    info = EsmModelInfo(conf_sp.plm)
    model = Header(
        in_size=info['dim'], hidden_size=conf_sp.rnn_hidden_size,
        num_layers=conf_sp.rnn_layers, num_classes=2, net_type=conf_sp.rnn_type).to(dev)
    model.load_state_dict(biolip_state)
    model.eval()

    test_df = pd.read_csv(
        f'data/biolip/by_metal/{metal}_test.txt', header=None, sep=',')
    val_df = pd.read_csv(
        f'data/biolip/by_metal/{metal}_val.txt', header=None, sep=',')
    train_df = pd.read_csv(
        f'data/biolip/by_metal/{metal}_train.txt', header=None, sep=',')
    df = pd.concat([test_df, val_df, train_df], ignore_index=True)
    df.columns = ['Seq ID', 'Sequence', 'X', 'Y', 'Z', 'Type']

    # consider only biological metal binding sites
    df = df[df['Type'] == 'Bio']

    total, positives, negatives = 0, 0, 0
    all_site_count, all_missing, num_proteins = 0, 0, 0
    missing_pdbids = []

    for (seqid, seq), subdf in (pbar := tqdm(df.groupby(by=['Seq ID', 'Sequence']), dynamic_ncols=True)):
        if (cache_dir / f"{seqid}.pt").exists():
            continue

        # collect ALL metal binding sites
        sites = [(row['X'], row['Y'], row['Z']) for i, row in subdf.iterrows()]

        # predict by sequence
        _, emb, _ = torch.load(
            Path(f'cache/biolip/{conf_sp.plm}/{seqid}.pt'), map_location=dev)
        with torch.no_grad():
            logits = model(emb.unsqueeze(0).float()).detach()

        prob = logits[0].softmax(-1)[..., 1]
        pred = prob >= conf_sp.detect_thresh
        min_sites = num_sites(seq)

        # clamp the number of sites to [min_sites, max_sites]
        if pred.sum() < min_sites:
            pred = torch.zeros_like(prob, dtype=torch.bool)
            pred[prob.argsort(descending=True)[:min_sites]] = True
        elif pred.sum() > conf_sp.max_sites:
            pred = torch.zeros_like(prob, dtype=torch.bool)
            pred[prob.argsort(descending=True)[:conf_sp.max_sites]] = True

        # load pdb file
        atoms = get_structure(CIFFile.read(f'data/cifs/{seqid[:4]}.cif'))[0]
        metal_atoms = atoms[atoms.res_name == metal]
        atoms = atoms[(atoms.hetero == False)]

        # it should be close to the metal binding sites
        NOS_coords, NOS_elem, NOS_probs, NOS_resnia = [], [], [], []
        backbone_atoms = backbone(atoms, seqid[4:])
        seq2res = build_map(seq, backbone_atoms)
        cmd_str = f'rei;fetch {seqid[:4]};remove not chain {seqid[4:]};remove hetatm and not name {metal};center;remove solvent;hide cartoon;show sticks;color green;color brown, metals;'

        for i in torch.nonzero(pred).flatten():
            nos = get_nos_atoms((
                i, prob.cpu().numpy(), seq2res, atoms, seqid[4:], metal, processed_stats, conf_sp.probe_res_thresh))
            if nos is not None:
                NOS_coords.append(torch.tensor(nos[0]).view(-1, 3))
                NOS_elem.extend(nos[1])
                NOS_probs.extend(nos[2])
                NOS_resnia.extend(nos[3])
                cmd_str += nos[4]

        ALL_coords = torch.tensor(atoms.coord)

        # detect the possible metal binding sites
        with torch.no_grad(), torch.amp.autocast('cuda'):
            possible_sites = get_probe(
                metal,
                processed_stats, NOS_coords, NOS_probs, NOS_resnia,
                NOS_elem, ALL_coords, conf_sp.probe_size, dev,
                lb, ub,
                conf_sp.probe_score_thresh, conf_sp.detect_thresh, conf_sp.max_offset,)
            possible_sites, kskp = torch.split(possible_sites, [3, 2], dim=-1)

        cmd_str += generate_pymol_script(possible_sites)

        sites = torch.tensor(sites).to(dev)
        # sites = torch.from_numpy(metal_atoms.coord).to(dev)

        vec = sites.unsqueeze(0) - possible_sites.unsqueeze(1)
        # dist of each probe to the nearest metal site
        dist = torch.norm(vec, dim=-1).min(dim=1)
        offset = vec[torch.arange(vec.shape[0]), dist.indices]
        label = dist.values <= conf_sp.max_offset

        missing = len(sites) - len(dist.indices[label].unique())
        all_site_count += len(sites)
        all_missing += missing

        if missing:
            missing_pdbids.append(seqid)

        if missing:  # True, missing:
            cmd_str += generate_pymol_script(sites, c='red', prefix=metal)
            import pymol
            pymol.finish_launching(['pymol', '-qc'])
            pymol.cmd.feedback("disable", "all", "everything")

            for subcmd in tqdm(segment_cmd(cmd_str, max_len=5000), leave=False):
                pymol.cmd.do(subcmd)
            pymol.cmd.do('set sphere_scale, 0.3')
            pymol.cmd.save(
                f'probes/{metal}/{seqid[:4]}_probes_{metal}.pse')
            print(f'\n\nseqid={seqid}, missing={missing}, total={len(sites)}')
            # exit(0)

        total += len(label)
        positives += label.sum().item()
        negatives += (~label).sum().item()
        num_proteins += 1
        pbar.set_postfix_str(
            f"pos: {positives}/{negatives}={positives/negatives:.4f}, miss: {all_missing}/{all_site_count}={all_missing/all_site_count:.4f}, probes/prot={total/num_proteins:.2f}")

        torch.save(
            (possible_sites, offset, kskp), cache_dir / f"{seqid}.pt")

        # if num_proteins > 10:
        #     break
    msg = f"[{metal}] total: {total}, positives: {positives}, negatives: {negatives}, missing: {all_missing}, all_site_count: {all_site_count}, num_proteins: {num_proteins}"
    logger.info(msg)

    with open('data/biolip/missing_seqids.csv', 'a') as f:
        f.write(
            f'{metal},{total},{positives},{negatives},{all_missing},{all_site_count},{num_proteins},{";".join(missing_pdbids)}\n')

    os.system('rm -f *.pse *.cif *.pdb')
