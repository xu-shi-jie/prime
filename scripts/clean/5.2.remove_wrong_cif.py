# Author: Shijie Xu
# Date: 2025-05-21
# Description: Remove incorrectly fixed cif files

import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np
import torch
import pickle
# fmt: off
import sys
sys.path.append('.')
from models.utils import safe_dist
# fmt: on
n_sample = 10
radius = 10.0
# remove incorrectly fixed cifs
if Path('data/biolip/cif_mates.pkl').exists():
    cif_mates = pickle.load(open('data/biolip/cif_mates.pkl', 'rb'))
else:
    cif_mates = {}


def check_valid(cif):
    if cif_mates.get(cif.stem, 'Not existed') == 'Success':
        return None

    try:
        atoms = get_structure(CIFFile.read(cif))[0]
    except Exception as e:
        return (cif.stem, str(e))

    indices = torch.randint(0, len(atoms), (n_sample,), device='cuda')
    coords = torch.tensor(atoms.coord, device='cuda')
    centers = coords[indices]
    mass = (torch.cdist(coords, centers) <= radius).sum() / n_sample
    volume = radius ** 3
    density = mass / volume

    if density > 2.0:
        return (cif.stem, 'Incorrect cif')
    elif density > 0.5:
        print(f'{cif}: {density:.2f}')
        exit(0)
    else:
        return (cif.stem, 'Success')


if __name__ == '__main__':
    files = list(Path('data/fixed_cifs/').glob('**/*.cif'))
    # use spawn
    torch.multiprocessing.set_start_method('spawn', force=True)

    results = []
    with mp.Pool(8) as pool:
        results = list(tqdm(
            pool.imap(check_valid, files),
            total=len(files), desc='Checking cif files'))
    # for cif in tqdm(files, desc='Checking cif files'):
    #     results.append(check_valid(cif))

    for r in results:
        if r is not None:
            cif, result = r
            cif_mates[cif] = result

    with open('data/biolip/cif_mates.pkl', 'wb') as f:
        pickle.dump(cif_mates, f)
    print('Done!')
