import gzip
from Bio.SeqIO import parse
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
# fmt: off
import sys
sys.path.append('.')
from models.pdb import cif_res
# fmt: on

fasta = parse("data/pretrain/pdb_seqres.txt", "fasta")
pdbids = [r.id[:4].lower() for r in fasta]
pdbids = list(set(pdbids))
ciffiles = [f"/database/mmCIF/{pdbid[1:3]}/{pdbid}.cif.gz" for pdbid in pdbids]
ciffiles = [f for f in ciffiles if Path(f).exists()]

with mp.Pool(mp.cpu_count()) as pool:
    cif_dict = list(tqdm(pool.imap(cif_res, ciffiles), total=len(ciffiles)))

cif_dict = dict(zip(pdbids, cif_dict))
fasta = parse("data/pretrain/pdb_seqres.txt", "fasta")
with open("data/pretrain/pdb_seqres_high_res.txt", "w") as f:
    for r in tqdm(fasta):
        pdbid = r.id[:4].lower()
        if pdbid in cif_dict:
            res, method = cif_dict[pdbid]
            if method == "X-RAY DIFFRACTION" and res is not None and res <= 2.0:
                f.write(f">{r.id}\n{r.seq}\n")
