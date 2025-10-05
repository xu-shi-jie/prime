from einops import repeat
from torch_scatter import scatter_max, scatter_mean
from scipy.spatial import KDTree
import re
import pandas as pd
from tqdm import tqdm
import line_profiler
from typing import List, Tuple, Dict, Any
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
import torch
from math import ceil
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import gzip
from typing import List
# fmt: off
import sys
sys.path.append('.')
from models.utils import remove_close_points_kdtree, pack_bit, safe_cdist_thr, scatter_medoid, unpack_bit, safe_filter, safe_dist
from models.dataset import _1to3
# fmt: on
# https://chem.libretexts.org/Courses/Mount_Royal_University/Chem_1201/Unit_2._Periodic_Properties_of_the_Elements/2.08%3A_Sizes_of_Atoms_and_Ions
# Ionic radius data from R. D. Shannon, “Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides,” Acta Crystallographica 32, no. 5 (1976): 751–767.
vdw = {
    'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'SE': 1.9, 'F': 1.47,
    'ZN': 0.74, 'CA': 1.00, 'MG': 0.72, 'MN': 0.83,
    'FE': 0.645, 'CU': 0.73, 'CO': 0.745, 'NI': 0.69,
    'CD': 0.95, 'K': 1.38,
    'FE2': 0.78, 'NA': 1.02,
    # Revised radii of the univalent Cu, Ag, Au and Tl cations https://journals.iucr.org/b/issues/2020/01/00/lo5064/index.html
    'CU1': 0.74,
    # https://www.sciencedirect.com/science/article/pii/S092145260500832X
    'MN3': 0.65,
    'AU': 1.37, 'AG': 1.15,
    # https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119961468.app7, Crystallography and Crystal Defects, Second Edition. Anthony Kelly and Kevin M. Knowles.
    '3CO': 0.61,
    'HG': 1.02,
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC2567809/ "Conversely, Pb2+ has a reported ionic radius of 1.12 Å – 1.19 Å for similar coordination. "
    'PB': 1.19,
    # https://pubs.rsc.org/en/content/articlepdf/2022/ra/d2ra02199d
    'TB': 1.04,
}
# # use openbabel to get the vdw radius
# obabel_vdw = pd.read_csv('models/element.txt', sep='\t', skiprows=33)
# vdw = {}
# for i, row in obabel_vdw.iterrows():
#     vdw[row['Symb'].upper()] = float(row['RVdW'])
# vdw['FE2'] = vdw['FE']
# vdw['CU1'] = vdw['CU']
# vdw['MN3'] = vdw['MN']
# vdw['3CO'] = vdw['CO']


def cif_res(cif_file):
    """return the resolution and method used to determine the structure"""
    file = gzip.open(cif_file, "rt")
    info = MMCIF2Dict(file)
    res, method = (
        info.get("_reflns.d_resolution_high", None),
        info.get("_exptl.method", None),
    )
    try:
        res = float(res[0])
    except:
        res = None
    return res, method[0]


def rand_rot(dev) -> torch.Tensor:
    # Generate random angles for rotations around z, y, and x axes
    angles = torch.rand(3, device=dev) * 2 * torch.pi

    # Precompute trigonometric values
    coses, sines = torch.cos(angles), torch.sin(angles)

    # Directly compute the combined rotation matrix
    R = torch.tensor([
        [coses[0] * coses[1], coses[0] * sines[1] * sines[2] - sines[0] *
         coses[2], coses[0] * sines[1] * coses[2] + sines[0] * sines[2]],
        [sines[0] * coses[1], sines[0] * sines[1] * sines[2] + coses[0] *
         coses[2], sines[0] * sines[1] * coses[2] - coses[0] * sines[2]],
        [-sines[1], coses[1] * sines[2], coses[1] * coses[2]]
    ])
    return R.to(dev)


def safe_min_dist_index(vec1: torch.Tensor, vec2: torch.Tensor, dim: int = 0, max_size: int = 10_000_000):
    """ compute the minimum distance between two vectors:

    vec1: (N, 3), N could be very very large, i.e., all atoms' coordinates in a large protein

    vec2: (M, 3), M are not very large, usually the coordinates of the binding sites

    return: (M, ) the minimum distance of each binding site to the protein
    """
    size1 = vec1.shape
    size2 = vec2.shape
    batch_size = ceil(max_size / size2[0])
    values, indices = [], []
    for i in range(0, size1[0], batch_size):
        dist = torch.cdist(vec1[i:i + batch_size], vec2)
        dist = dist.min(dim=dim)
        values.append(dist.values)
        indices.append(dist.indices)
    return torch.cat(values), torch.cat(indices)


def num_sites(seq):
    """ Estimate the number of candidate sites for a given sequence length."""
    x = len(seq)
    return round(1.8218378213155169 * x**0.6250181898331563 + 0.06288782372587086)


num_sites_ranges = {
    'ZN': (-1, 10000),
    'CA': (100, 10000),
    'MG': (100, 10000),
    'MN': (-1, 10000),
    'FE': (100, 10000),
    'FE2': (-1, 10000),
    'CO': (100, 10000),
    'CU': (100, 10000),
}


def generate_pymol_script(possible_sites, prefix='pr', c: str = 'blue'):
    cmd = ''
    for i, pos in enumerate(possible_sites):
        cmd += f"pseudoatom {prefix}{i},pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}];color {c},{prefix}{i};"
    return cmd


def resnia2res(resnia):
    res, at = [], []
    for x in resnia:
        x1, x2, x3 = x.split(',')
        res.append(x1+x2)
        at.append(x3)
    return res, at


# @line_profiler.profile
def collect_interests(interests, dist_nos, positions, nos_probs, nos_resnia, samples, ks_thresh, kp_thresh):
    """ Collect the binding sites based on the distance matrix """
    keys, indices = torch.unique(
        dist_nos.contiguous(), dim=1, return_inverse=True)
    max_key = indices.max().item()
    nos_resnia = np.array(nos_resnia)
    keys = unpack_bit(keys.T, len(nos_probs)).bool()
    nos_samples_probs = torch.stack([samples, nos_probs])
    positions = positions[interests]

    clustered_positions = scatter_mean(
        positions, indices, dim=0, dim_size=max_key + 1)
    # clustered_positions, _ = scatter_medoid(
    #     positions, indices, dim_size=max_key + 1)

    scores = nos_samples_probs.unsqueeze(1) * keys
    max_scores = scores.max(dim=-1).values
    max_scores = scores.sum(dim=-1)
    # mask = (max_scores[0] > ks_thresh) & (max_scores[1] > kp_thresh)
    # return torch.cat([clustered_positions, max_scores.T.contiguous()], dim=-1)[mask]
    return torch.cat([clustered_positions, max_scores.T.contiguous()], dim=-1)


@line_profiler.profile
def get_probe(metal, statistics, nos_coords, nos_probs, nos_resnia, nos_elem, all_coords, probe_size, dev, lb, ub, ks_thresh, kp_thresh, min_dist):
    """ Get the probe positions for the binding sites
    Args:
        metal (str): The metal type, e.g., 'ZN', 'CA'.
        statistics (dict): The statistics for the metal.
        nos_coords (list): List of coordinates of the N, O, S atoms.
        nos_probs (list): List of probabilities for the N, O, S atoms.
        nos_resnia (list): List of residue names and indices for the N, O, S atoms.
        nos_elem (list): List of elements for the N, O, S atoms.
        all_coords (torch.Tensor): All coordinates in the protein.
        probe_size (float): Size of the probe.
        dev (torch.device): Device to use for computation.
        lb (float): Lower bound for distance filtering.
        ub (float): Upper bound for distance filtering.
        thresh (float): Threshold for detection.
        min_dist (float): Minimum distance between probes.
    Returns:
        torch.Tensor: The positions of the probes.
    """
    positions = []

    for i, _nos in tqdm(enumerate(nos_coords), leave=False, desc='Generating probes'):
        min_x, min_y, min_z = _nos.min(axis=0).values - 5
        max_x, max_y, max_z = _nos.max(axis=0).values + 5
        size = max_x - min_x, max_y - min_y, max_z - min_z
        xx, yy, zz = torch.meshgrid(
            torch.linspace(min_x, max_x, ceil(size[0] / probe_size)),
            torch.linspace(min_y, max_y, ceil(size[1] / probe_size)),
            torch.linspace(min_z, max_z, ceil(size[2] / probe_size)),
            indexing='ij'
        )
        xxyyzz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        if 0 < i < len(nos_coords) - 1:  # neighboring residues are likely to be close
            _nos = torch.cat([_nos, nos_coords[i + 1]])
        elif i == 0:
            _nos = torch.cat([_nos, nos_coords[i + 1]])
        elif i == len(nos_coords) - 1:
            _nos = torch.cat([_nos, nos_coords[i - 1]])

        dist = torch.cdist(xxyyzz, _nos)
        dist = dist.min(dim=1).values
        mask = (dist >= lb) & (dist <= ub)
        positions.append(xxyyzz[mask])

    positions = torch.cat(positions).to(dev)
    nos_coords = torch.cat(nos_coords).to(dev).to(torch.bfloat16)
    nos_probs = torch.tensor(nos_probs).to(dev).to(torch.bfloat16)
    all_coords = all_coords.to(dev)

    threshes = []
    samples = []
    for resnia in nos_resnia:
        resn, resi, an = resnia.split(',')
        key = f'{resn},{an},{metal}'
        if key not in statistics:
            threshes.append((lb, ub))
            samples.append(0)
        else:
            threshes.append((statistics[key][0], statistics[key][1]))
            samples.append(statistics[key][2])

    threshes = torch.tensor(threshes).to(dev)
    samples = torch.tensor(samples).to(dev)

    # should not be too far to the N, O, S atoms of binding residues
    dist_nos, interests = safe_filter(
        nos_coords, positions, threshes, all_coords, lb, max_size=1000000000)

    binding_sites = collect_interests(
        interests, dist_nos, positions, nos_probs, nos_resnia, samples, ks_thresh, kp_thresh)

    sort_idx = torch.argsort(binding_sites[:, 3], descending=True)
    binding_sites = binding_sites[sort_idx]

    thr = min_dist / 2

    dist = safe_cdist_thr(binding_sites[:, :3], thr)
    indices = torch.zeros((3000,), dtype=torch.long, device=dev)
    num_probes = 0
    for i in range(binding_sites.shape[0]):
        if num_probes:
            _dist = dist[i, indices[:num_probes]]
            if torch.any(_dist):
                continue
        if num_probes >= indices.shape[0]:
            indices = torch.cat(
                [indices, torch.zeros(3000, dtype=torch.long).to(dev)])
        indices[num_probes] = i
        num_probes += 1

    return binding_sites[indices[:num_probes]]


# @line_profiler.profile
def get_nos_atoms(params):
    seq_pos, prob, seq2res, \
        atoms, chain_id, metal, processed_stats, \
        probe_res_thresh = params

    p = prob[seq_pos].item()
    res = seq2res[seq_pos]
    if res is None:
        return None

    resn, resi = _1to3.get(res[0], 'UNK'), int(res[1:])
    if resn != 'UNK':
        at = atoms[
            (atoms.res_id == resi) &
            (atoms.res_name == resn) &
            (atoms.chain_id == chain_id)]  # get residue atoms
        res_label = f'{chain_id}_{resn}{resi}'
        cmd_str = f'select {res_label},resi {resi} and chain {chain_id};color orange,{res_label};'
    else:
        at = atoms[
            (atoms.res_id == resi) &
            (~np.isin(atoms.res_name, list(_1to3.values()))) &
            (atoms.chain_id == chain_id)]
    nos_key = [
        f'{x.res_name},{x.atom_name},{metal}' for x in at]
    nos_idx = []
    for i, key in enumerate(nos_key):
        if processed_stats.get(key, (0, 0, 0))[2] >= probe_res_thresh:
            nos_idx.append(True)
        else:
            nos_idx.append(False)
    nos_idx = np.array(nos_idx)

    if nos_idx.sum():
        return at[nos_idx].coord, \
            list(at[nos_idx].element), \
            [p] * nos_idx.sum(), \
            [f'{resn},{resi},{i}' for i in at[nos_idx].atom_name], \
            cmd_str
    else:
        return None
