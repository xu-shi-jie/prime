import pandas as pd
from Bio.SeqIO import parse
import argparse
from pathlib import Path
import pickle
import gzip
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite import InvalidFileError
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from loguru import logger
import requests
import shutil
# fmt: off
import sys
sys.path.append('.')
from models.dataset import read_biolip, _1to3, metals, _lig2elem, _33, considered_metals
from models.pdb import vdw
# fmt: on
DEBUG = False
torch.multiprocessing.set_start_method('spawn', force=True)


def get_dev():
    rank = mp.current_process()._identity
    if len(rank) == 0:
        return torch.device('cuda', index=0)
    return torch.device('cuda', index=mp.current_process(
    )._identity[0] % torch.cuda.device_count())


def extract_annotations(row):
    pdbid, val = row
    annots = []
    try:
        atoms = get_structure(CIFFile.read(f'data/cifs/{pdbid}.cif'))[0]
    except InvalidFileError:
        return []

    allow_error = 0.5
    for chain, lig_name, seq, lig_chain, binding_res in val:
        for res in binding_res.split():
            if lig_name not in considered_metals:
                continue

            resn, resi = res[0], res[1:]
            if resi.isdigit():
                resi = int(resi)
                ins_code = ''
            else:
                try:
                    resi = int(resi[:-1])
                    ins_code = resi[-1]
                except:
                    # for example: 4zn3, chain B, resi -5 is negative,
                    # we do not deal with this case currently
                    continue
            resn = _33.get(resn, 'X')
            if resn == 'X':
                atom = atoms[
                    (atoms.res_id == resi) &
                    (atoms.ins_code == ins_code) &
                    (atoms.chain_id == chain) &
                    (atoms.element != 'H')
                ]
            else:
                atom = atoms[
                    np.isin(atoms.res_name, resn) &
                    (atoms.res_id == resi) &
                    (atoms.ins_code == ins_code) &
                    (atoms.chain_id == chain) &
                    (atoms.element != 'H')
                ]
            if len(atom) == 0:
                continue
            lig_atom = atoms[
                (atoms.chain_id == lig_chain) &
                (atoms.res_name == lig_name) &
                (atoms.element == _lig2elem.get(lig_name, lig_name)) &
                (atoms.atom_name == _lig2elem.get(lig_name, lig_name))
            ]
            if len(lig_atom) == 0:
                continue
            dev = get_dev()
            dist = torch.cdist(
                torch.tensor(atom.coord).to(dev),
                torch.tensor(lig_atom.coord).to(dev))
            min_dist, min_idx = torch.min(dist, dim=0)
            for idx_l in range(len(lig_atom)):
                idx_p = min_idx[idx_l].item()
                if vdw[atom[idx_p].element] + vdw[lig_atom[idx_l].res_name] + allow_error >= min_dist[idx_l]:
                    annots.append(
                        [pdbid+chain, seq, lig_atom[idx_l].coord, lig_name, 'Bio'])

    # remove duplicates binding sites
    coords = [a[2] for a in annots]
    coords = np.array(coords)
    coords, idx = np.unique(coords, axis=0, return_index=True)
    annots = [annots[i] for i in idx]

    return annots


def check_duplicate(this, ds):
    pdbid_chain, seq, coord = this
    for _pdbid_chain, _seq, _coord, dt in ds:
        if _pdbid_chain == pdbid_chain and seq == _seq and np.allclose(coord, _coord):
            return True
    return False


if __name__ == '__main__':
    biolip = read_biolip("data/biolip/BioLiP.txt")
    biolip.drop_duplicates(subset=[
        'PDB ID',
        'Receptor chain',
        'Ligand ID',
        'Ligand chain',
        'Receptor sequence',
        'Binding site residues (PDB)'
    ], inplace=True)
    logger.info(f'Removed duplicates: {len(biolip)} entries')
    train_ids = [r.id for r in parse(
        "data/biolip/seq/seq_train.fasta", "fasta")]
    val_ids = [r.id for r in parse(
        "data/biolip/seq/seq_val.fasta", "fasta")]
    test_ids = [r.id for r in parse(
        "data/biolip/seq/seq_test.fasta", "fasta")]

    Path('data/biolip/rows.pkl').unlink(missing_ok=True)
    Path('data/cifs').mkdir(exist_ok=True, parents=True)
    if Path('data/biolip/rows.pkl').exists():
        rows = pickle.load(open('data/biolip/rows.pkl', 'rb'))
    else:
        rows = {}
        for i, row in tqdm(biolip.iterrows(), total=len(biolip), desc='Processing rows'):
            pdbid = row["PDB ID"]
            chain = row["Receptor chain"]
            seqid = pdbid + chain

            if seqid not in train_ids + val_ids + test_ids:
                continue

            lig_name = row["Ligand ID"]
            lig_chain = row["Ligand chain"]
            seq = row["Receptor sequence"]
            binding_res = row["Binding site residues (PDB)"]
            d = rows.get(pdbid, [])

            d.append((chain, lig_name, seq, lig_chain, binding_res))
            rows[pdbid] = d

        rows = list(rows.items())
        pickle.dump(rows, open('data/biolip/rows.pkl', 'wb'))

    logger.info(f'Filtered {len(rows)} rows')
    rows = sorted(rows, key=lambda x: len(x[1]), reverse=True)

    results = []

    # downaload all cif files
    for row in tqdm(rows, total=len(rows), desc='Downloading cif files'):
        pdbid = row[0]
        if Path(f'data/cifs/{pdbid}.cif').exists():
            continue
        elif Path(f'/database/mmCIF/{pdbid[1:3]}/{pdbid}.cif.gz').exists():
            # unzip the file
            with open(f'data/cifs/{pdbid}.cif', 'wb') as f, \
                    gzip.open(f'/database/mmCIF/{pdbid[1:3]}/{pdbid}.cif.gz', 'rb') as gz_f:
                f.write(gz_f.read())
        else:
            url = f'https://files.rcsb.org/view/{pdbid}.cif'
            r = requests.get(url)
            if r.status_code == 200:
                with open(f'data/cifs/{pdbid}.cif', 'wb') as f:
                    f.write(r.content)
            else:
                raise ValueError(f'Cannot download the cif file for {pdbid}')

    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(extract_annotations, rows), total=len(rows), desc='Extracting annotations'))
    # for row in tqdm(rows, total=len(rows), desc='Extracting annotations'):
    #     results.append(extract_annotations(row))

    flatten_results = []
    for annots in tqdm(results, desc='Flattening results'):
        flatten_results.extend(annots)

    shutil.rmtree('data/biolip/by_metal', ignore_errors=True)
    Path('data/biolip/by_metal').mkdir(exist_ok=True, parents=True)

    data_by_metal = {}
    for pdbid_chain, seq, coord, atom_name, dt in tqdm(flatten_results, desc='Processing annotations'):
        metal = data_by_metal.get(atom_name, [])
        metal.append([pdbid_chain, seq, coord, dt])
        data_by_metal[atom_name] = metal

    for metal, data in data_by_metal.items():
        if metal not in considered_metals:
            continue

        train, test, val = [], [], []
        for pdbid_chain, seq, coord, dt in tqdm(data, desc=f'Processing {metal}', leave=False):
            if pdbid_chain in train_ids and not check_duplicate((pdbid_chain, seq, coord), train):
                train.append([pdbid_chain, seq, coord, dt])
            elif pdbid_chain in test_ids and not check_duplicate((pdbid_chain, seq, coord), test):
                test.append([pdbid_chain, seq, coord, dt])
            elif pdbid_chain in val_ids and not check_duplicate((pdbid_chain, seq, coord), val):
                val.append([pdbid_chain, seq, coord, dt])
        logger.info(
            f'{metal}: train={len(train)}, test={len(test)}, val={len(val)}')
        with open(f'data/biolip/by_metal/{metal}_train.txt', 'w') as f:
            for pdbid_chain, seq, coord, dt in train:
                f.write(
                    f'{pdbid_chain},{seq},{coord[0]},{coord[1]},{coord[2]},{dt}\n')
        with open(f'data/biolip/by_metal/{metal}_test.txt', 'w') as f:
            for pdbid_chain, seq, coord, dt in test:
                f.write(
                    f'{pdbid_chain},{seq},{coord[0]},{coord[1]},{coord[2]},{dt}\n')
        with open(f'data/biolip/by_metal/{metal}_val.txt', 'w') as f:
            for pdbid_chain, seq, coord, dt in val:
                f.write(
                    f'{pdbid_chain},{seq},{coord[0]},{coord[1]},{coord[2]},{dt}\n')
