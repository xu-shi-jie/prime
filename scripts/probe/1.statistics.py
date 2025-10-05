from pathlib import Path
from math import ceil
import multiprocessing as mp
from biotite.structure.io.pdbx import CIFFile, get_structure
import pickle
from numba import njit, prange
import torch
from tqdm import tqdm
from loguru import logger
import numpy as np
# fmt: off
import sys
sys.path.append('.')
from models.dataset import read_biolip, metals, _lig2elem, considered_metals
from models.pdb import vdw, safe_dist, safe_min_dist_index
# fmt: on
torch.multiprocessing.set_start_method('spawn', force=True)


def get_dev():
    idx = mp.current_process()._identity
    idx = idx[0] if len(idx) else 0
    return torch.device('cuda', idx % torch.cuda.device_count())
    # return torch.device('cpu')


def get_dist(at1, at2):
    dev = get_dev()
    return torch.cdist(
        torch.tensor(at1.coord).to(dev).unsqueeze(0),
        torch.tensor(at2.coord).to(dev).unsqueeze(0)).item()


def get_angle(at1, at2, at3):
    """ return the angle of at1-at2-at3 """
    dev = get_dev()
    at1 = torch.tensor(at1.coord).to(dev)
    at2 = torch.tensor(at2.coord).to(dev)
    at3 = torch.tensor(at3.coord).to(dev)
    v1 = (at1 - at2).reshape(-1)
    v2 = (at3 - at2).reshape(-1)
    return torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))).item()


def extract_statistic(params):
    pdbid, val = params

    distances = []

    try:
        atoms = get_structure(CIFFile.read(f'data/cifs/{pdbid}.cif'))[0]
    except:
        return distances

    nos_atoms = atoms[
        np.isin(atoms.element, ['N', 'O', 'S'])]
    dev = get_dev()

    nos_coords = torch.tensor(nos_atoms.coord).to(dev)
    max_r = torch.zeros(len(nos_atoms), device=dev)+0.5
    for elem in 'NOS':
        mask = torch.from_numpy(nos_atoms.element == elem).to(dev)
        max_r[mask] += vdw[elem]

    for chain, lig_name, lig_chain, binding_res in val:
        if lig_name not in considered_metals:
            continue
        lig_atom = atoms[
            (atoms.chain_id == lig_chain) &
            (atoms.res_name == lig_name) &
            (atoms.element == _lig2elem.get(lig_name, lig_name)) &
            (atoms.atom_name == _lig2elem.get(lig_name, lig_name))
        ]
        if len(lig_atom) == 0:
            continue

        values, _ = safe_min_dist_index(
            nos_coords, torch.tensor(lig_atom.coord).to(dev), dim=1)
        mask = values < max_r + vdw[lig_name]

        distances.extend([(
            pdbid,
            lig_name,
            nos_atoms.res_name[idx.item()],
            nos_atoms.atom_name[idx.item()],
            nos_atoms.element[idx.item()],
            v.item()) for idx, v in zip(torch.nonzero(mask, as_tuple=True)[0], values[mask])])

    return distances


if __name__ == '__main__':
    biolip = read_biolip('data/biolip/BioLiP.txt')
    trains = open('data/biolip/seq/seq_train.fasta').read().splitlines()
    vals = open('data/biolip/seq/seq_val.fasta').read().splitlines()
    train_val_pdbids = list(set([
        l[1:5] for l in trains + vals if l.startswith('>')]))
    # tests = open('data/biolip/seq/seq_test.fasta').read().splitlines()
    # test_pdbids = list(set([
    #     l[1:5] for l in tests if l.startswith('>')]))

    Path('data/biolip/statistics.pkl').unlink(missing_ok=True)
    if Path('data/biolip/statistics.pkl').exists():
        rows = pickle.load(open('data/biolip/statistics.pkl', 'rb'))
    else:
        rows = {}
        for i, row in tqdm(biolip.iterrows(), total=len(biolip), desc='Processing rows'):
            pdbid = row['PDB ID']
            # if pdbid in test_pdbids:
            if pdbid not in train_val_pdbids:
                continue
            chain = row['Receptor chain']
            lig_name = row['Ligand ID']
            if lig_name not in considered_metals:
                continue
            lig_chain = row['Ligand chain']
            binding_res = row['Binding site residues (PDB)']
            rows.setdefault(pdbid, []).append(
                [chain, lig_name, lig_chain, binding_res])
        rows = list(rows.items())
        pickle.dump(rows, open('data/biolip/statistics.pkl', 'wb'))

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(extract_statistic, rows), total=len(rows), desc='Extracting statistics'))

    # results = []
    # for row in tqdm(rows, desc='Extracting statistics'):
    #     results.append(extract_statistic(row))

    flatten_results = []
    for r in results:
        flatten_results.extend(r)

    logger.info(f'Number of results: {len(flatten_results)}')
    dists = {}
    with open('data/biolip/statistics_distance.pkl', 'wb') as f:
        for pdbid, m, rn, an, e, v in tqdm(flatten_results):
            dists.setdefault((m, e), []).append((pdbid, rn, an, v))
        pickle.dump(dists, f)
