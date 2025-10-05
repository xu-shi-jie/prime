from loguru import logger
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np


# fmt: off
import sys
sys.path.append('.')
from models.pdb import vdw
from models.dataset import lig2name, considered_metals, _3to1
# fmt: on


def std_lh(v):
    mu, sigma = np.mean(v), np.std(v)
    return mu - 3.0 * sigma, mu + 3.0 * sigma


def iqr_lh(v, delta=1.5):
    q1, q3 = np.percentile(v, [25, 75])
    # iqr = max(q3-q1, 0.5)
    iqr = max(q3 - q1, 0.5)
    lb, ub = q1 - delta * iqr, q3 + delta * iqr
    return lb, ub


if __name__ == '__main__':
    data = pickle.load(open('data/biolip/statistics_distance.pkl', 'rb'))
    data = dict(sorted(data.items(), key=lambda x: x[0]))
    df = []
    for k, _v in tqdm(data.items(), leave=False):
        if k[1] == 'C' or k[0] not in considered_metals:
            continue
        for pdbid, rn, an, v in _v:
            if rn in _3to1:
                df.append(
                    (k[0], k[1], f'{lig2name[k[0]]}-{lig2name[k[1]]}', v, pdbid, rn, an, len(_v)))

    df = pd.DataFrame(df, columns=['Metal', 'Atom', 'Pair',
                                   'Distance', 'PDB ID', 'Residue name', 'Atom name', 'Num'])
    residue_atoms = {}
    for i, row in tqdm(df.iterrows(), leave=False, total=len(df)):
        residue_atoms.setdefault(
            (row['Residue name'], row['Atom name'], row['Metal']), []).append(row['Distance'])

    exported = []
    for k, v in residue_atoms.items():
        l, h = iqr_lh(v)
        exported.append((*k, l, h, len(v)))
    exported = pd.DataFrame(
        exported, columns=['Residue name', 'Atom name', 'Metal', 'L', 'H', 'Num'])
    exported.to_csv('data/biolip/statistics_distance_residue.csv', index=False)
    print(f'Min L: {exported["L"].min()}, Max H: {exported["H"].max()}')
