# Date: 2025-05-12
# Author: Shijie Xu
# Description: Filter the PDBs in the BioLiP dataset without necleic acid ligands

# fmt: off
import gzip
import multiprocessing as mp
from math import ceil, floor
import random
from biotite.structure.io.pdbx import CIFFile, get_structure
import shutil
from Bio.SeqIO import parse
import shlex
import numpy as np
import subprocess
import uuid
from tqdm import tqdm
import pandas as pd
import argparse
from pathlib import Path
import os
import sys
sys.path.append('.')
from models.dataset import read_biolip, considered_metals
# fmt: on


def check_available(pdbid):
    try:
        atoms = get_structure(CIFFile.read(gzip.open(
            f'/database/mmCIF/{pdbid[1:3]}/{pdbid}.cif.gz', 'rt')))[0]
    except:
        return None

    if np.isin(atoms.res_name, ['A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT']).any():
        return None

    return pdbid


if __name__ == '__main__':
    biolip = read_biolip('data/biolip/BioLiP.txt')
    available_pdbs = set()
    pdbids = np.unique(biolip['PDB ID'])
    if not Path('data/biolip/available_pdbs.txt').exists():
        available_pdbs = []
    else:
        available_pdbs = open(
            'data/biolip/available_pdbs.txt').read().splitlines()
    pdbids = set(pdbids) - set(available_pdbs)
    with mp.Pool(mp.cpu_count()) as pool:
        l = list(tqdm(pool.imap_unordered(
            check_available, pdbids), total=len(pdbids), desc='Checking PDBs'))

    with open('data/biolip/available_pdbs.txt', 'w') as f:
        for pdbid in l + available_pdbs:
            if pdbid is not None:
                pdbid = pdbid.strip()
                f.write(f'{pdbid}\n')
