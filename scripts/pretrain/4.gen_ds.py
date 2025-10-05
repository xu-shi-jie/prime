from pathlib import Path
import random
from Bio.SeqIO import parse
import gzip
from tqdm import tqdm
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np
import multiprocessing as mp
# fmt: off
import sys
sys.path.append('.')
from models.dataset import _3to1
# fmt: on

# pdbids = open("data/pretrain/train_val_ids.txt").read().splitlines()
pdbids = [
    r.id[:4].lower() for r in parse("data/pretrain/pdb_seqres_0.3.fasta", "fasta")
]
pdbids = list(set(pdbids))
print(f"Nonredundant PDBs: {len(pdbids)}")
# pdbids = [pdbid for pdbid in pdbids if Path(f"data/fixed_mmCIF/{pdbid}.cif").exists()]
# extract cif
cif_dir = '/database/mmCIF'
new_pdbids = []


def check_cif(pdbid):
    try:
        file = gzip.open(f'{cif_dir}/{pdbid[1:3]}/{pdbid}.cif.gz', 'rt')
        with open(f"data/cifs/{pdbid}.cif", "w") as f:
            f.write(file.read())
        atoms = get_structure(CIFFile.read(f"data/cifs/{pdbid}.cif"))[0]
        atoms = atoms[np.isin(atoms.res_name, list(
            _3to1.keys())) & (atoms.atom_name == 'CA')]
        if len(atoms):
            return True
    except:
        return False
    return False


# for pdbid in tqdm(pdbids, desc="Extracting CIFs"):
#     if check_cif(pdbid):
#         new_pdbids.append(pdbid)
with mp.Pool(mp.cpu_count()) as pool:
    valid_list = list(tqdm(pool.imap(check_cif, pdbids), total=len(pdbids)))
    new_pdbids = [pdbid for pdbid, valid in zip(pdbids, valid_list) if valid]
pdbids = new_pdbids

random.seed(42)
random.shuffle(pdbids)

with open("data/pretrain/val_ids.txt", "w") as f:
    f.write("\n".join(pdbids[:1000]))

with open("data/pretrain/train_ids.txt", "w") as f:
    f.write("\n".join(pdbids[1000:]))

print(f"Number of training samples: {len(pdbids[1000:])}")
print(f"Number of validation samples: {len(pdbids[:1000])}")
print(f"Number of total samples: {len(pdbids)}")
