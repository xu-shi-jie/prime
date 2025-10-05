import enum
from functools import partial
import gzip
from pathlib import Path
import random
import requests
import torch
from loguru import logger
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from biotite.structure.io.pdbx import CIFFile, get_structure
from models.plm import get_model
import lightning as L
from argparse import Namespace as Args
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

from models.utils import shorten_path
from typing import List
from torch.utils.data import Sampler

metals = open("data/biolip/ions.txt").read().splitlines()
metal2token = {m: i for i, m in enumerate(metals)}
metal2token.update({"UNK": len(metals)})
token2metal = {i: m for m, i in metal2token.items()}
aa_str = "ACDEFGHIKLMNPQRSTVWYX"
res2token = {aa: i for i, aa in enumerate(aa_str)}
_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
_1to3 = {v: k for k, v in _3to1.items()}
_33 = {
    # note that biolip use different 3- to 1-letter mapping from
    # https://www.ebi.ac.uk/pdbe-srv/pdbechem/chemicalCompound/show/ALA
    # for exampe, TPQ -> A, instead of Y
    'A': ['ALA', 'TPQ', 'CSD'],
    'C': ['CYS', 'CSX', 'OCS', 'SMC'],
    'D': ['ASP'],
    'E': ['GLU', 'CGU'],
    'F': ['PHE'],
    'G': ['GLY'],
    'H': ['HIS', 'HIC', 'NEP'],
    'I': ['ILE'],
    'K': ['LYS', 'KCX', 'MLY'],
    'L': ['LEU'],
    'M': ['MET', 'MSE'],
    'N': ['ASN'],
    'P': ['PRO'],
    'Q': ['GLN'],
    'R': ['ARG'],
    'S': ['SER', 'SEP'],
    'T': ['THR', 'TPO'],
    'V': ['VAL'],
    'W': ['TRP'],
    'Y': ['TYR',],
}
atom2token = {
    "C": 0,
    "CA": 1,
    "CB": 2,
    "CD": 3,
    "CD1": 4,
    "CD2": 5,
    "CE": 6,
    "CE1": 7,
    "CE2": 8,
    "CE3": 9,
    "CG": 10,
    "CG1": 11,
    "CG2": 12,
    "CH2": 13,
    "CZ": 14,
    "CZ2": 15,
    "CZ3": 16,
    "N": 17,
    "ND1": 18,
    "ND2": 19,
    "NE": 20,
    "NE1": 21,
    "NE2": 22,
    "NH1": 23,
    "NH2": 24,
    "NZ": 25,
    "O": 26,
    "OD1": 27,
    "OD2": 28,
    "OE1": 29,
    "OE2": 30,
    "OG": 31,
    "OG1": 32,
    "OH": 33,
    "OXT": 34,
    "SD": 35,
    "SG": 36,
    "H": 37,
    "H2": 38,
    "H3": 39,
    "HA": 40,
    "HA2": 41,
    "HA3": 42,
    "HB": 43,
    "HB1": 44,
    "HB2": 45,
    "HB3": 46,
    "HD1": 47,
    "HD11": 48,
    "HD12": 49,
    "HD13": 50,
    "HD2": 51,
    "HD21": 52,
    "HD22": 53,
    "HD23": 54,
    "HD3": 55,
    "HE": 56,
    "HE1": 57,
    "HE2": 58,
    "HE21": 59,
    "HE22": 60,
    "HE3": 61,
    "HG": 62,
    "HG1": 63,
    "HG11": 64,
    "HG12": 65,
    "HG13": 66,
    "HG2": 67,
    "HG21": 68,
    "HG22": 69,
    "HG23": 70,
    "HG3": 71,
    "HH": 72,
    "HH11": 73,
    "HH12": 74,
    "HH2": 75,
    "HH21": 76,
    "HH22": 77,
    "HZ": 78,
    "HZ1": 79,
    "HZ2": 80,
    "HZ3": 81,
}
# elem2token = {"C": 0, "N": 1, "O": 2, "S": 3, "H": 4, "P": 5}
elem2token = {"C": 0, "N": 1, "O": 2, "S": 3}
token2elem = {i: e for e, i in elem2token.items()}
_lig2elem = {
    # map ligand name to element, e.g. CU1 (1+) -> CU
    'CU1': 'CU',
    'FE2': 'FE',
    'IOD': 'I',
    'MN3': 'MN',
    '3CO': 'CO',
    'IR3': 'IR',
    '3NI': 'NI',
    'OS4': 'OS',
}


def read_biolip(p: str):
    """ read biolip dataset as a pandas dataframe """
    biolip = pd.read_csv(p, sep="\t", low_memory=False,
                         header=None, keep_default_na=False, na_values=None)
    # https://zhanggroup.org/BioLiP/download/readme.txt
    biolip.columns = [
        "PDB ID",
        "Receptor chain",
        "Resolution",
        "Binding site number code",
        "Ligand ID",
        "Ligand chain",
        "Ligand serial number",
        "Binding site residues (PDB)",
        "Binding site residues (Re-numbered)",
        "Catalytic site residues (PDB)",
        "Catalytic site residues (Re-numbered)",
        "EC number",
        "GO terms",
        "Binding affinity (Literature)",
        "Binding affinity (MOAD)",
        "Binding affinity (PDBbind-CN)",
        "Binding affinity (BindingDB)",
        "UniProt ID",
        "PubMed ID",
        "Residue sequence number of the ligand",
        "Receptor sequence",
    ]
    logger.info(f"Loaded BioLiP dataset with {len(biolip)} entries")
    return biolip


lig2name = {
    'ZN': 'Zn$^{2+}$',
    'CA': 'Ca$^{2+}$',
    'MG': 'Mg$^{2+}$',
    'MN': 'Mn$^{2+}$',
    'FE': 'Fe$^{3+}$',
    'CU': 'Cu$^{2+}$',
    'FE2': 'Fe$^{2+}$',
    'CO': 'Co$^{2+}$',
    'CU1': 'Cu$^{+}$',
    'K': 'K$^{+}$',
    'NA': 'Na$^{+}$',
    'NI': 'Ni$^{2+}$',
    'CD': 'Cd$^{2+}$',
    'MN3': 'Mn$^{3+}$',
    'AU': 'Au$^{+}$',
    'IOD': 'I$^{-}$',
    'AG': 'Ag$^{+}$',
    '3CO': 'Co$^{3+}$',
    'HG': 'Hg$^{2+}$',
    'PB': 'Pb$^{2+}$',
    'TB': 'Tb$^{3+}$',
    'LA': 'La$^{3+}$',

    'C': 'C',
    'N': 'N',
    'O': 'O',
    'S': 'S',
    'H': 'H',
    'P': 'P',
}
ligrank = {m: rank for rank, m in enumerate(lig2name.keys())}

transition_metals = [
    'ZN', 'MN', 'FE', 'CU', 'CO', 'FE2', 'CU1', 'NI', 'CD', 'MN3', 'AU', 'AG', '3CO', 'HG', 'PB', 'TB',
]
considered_metals = [
    'ZN', 'CA', 'MG', 'MN', 'FE', 'CU', 'FE2', 'CO', 'NA', 'CU1', 'K', 'NI', 'CD', 'HG',]
metal2elem = {
    'ZN': 'ZN', 'CA': 'CA', 'MG': 'MG', 'MN': 'MN', 'FE': 'FE', 'CU': 'CU',
    'FE2': 'FE', 'CO': 'CO', 'CU1': 'CU', 'K': 'K', 'NA': 'NA', 'NI': 'NI',
    'CD': 'CD', 'MN3': 'MN', 'AU': 'AU', 'AG': 'AG', '3CO': 'CO', 'HG': 'HG',
    'PB': 'PB', 'TB': 'TB', 'LA': 'LA',
}

abundants = ['ZN', 'CA', 'MG', 'MN', 'FE', 'CU']

all_elements = ['H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE',
                'NA', 'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR', 'K', 'CA',
                'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
                'GA', 'GE', 'AS', 'SE', 'BR', 'KR', 'RB', 'SR', 'Y', 'ZR',
                'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN',
                'SB', 'TE', 'I', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND',
                'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB',
                'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG',
                'TL', 'PB', 'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH',
                'PA', 'U', 'NP', 'PU', 'AM', 'CM', 'BK', 'CF', 'ES', 'FM',
                'MD', 'NO', 'LR', 'RF', 'DB', 'SG', 'BH', 'HS', 'MT', 'DS',
                'RG', 'CN', 'FL', 'LV', 'TS', 'OG']

all_elem2token = {e: i for i, e in enumerate(all_elements)}
all_token2elem = {i: e for e, i in all_elem2token.items()}


def build_map(seq, backbone):
    """ 
    Due to the inconsistency of residue numbering in PDB files, we need to map sequence position to atom position 
    """
    chain_seq = ''.join([_3to1.get(r, 'X') for r in backbone.res_name])
    chain_seqi = [str(r) for r in backbone.res_id]
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    align = aligner.align(seq, chain_seq)[0]
    indices = []
    i, j = 0, 0
    gap_head = True
    seq1, seq2 = align
    for s, c in zip(seq1, seq2):
        if s != '-':
            if gap_head:
                gap_head = False  # remove the gap at the beginning
            if c != '-':
                indices.append(chain_seq[j] + chain_seqi[j])
            else:
                indices.append(None)

        if s != '-' and not gap_head:
            i += 1
        if c != '-':
            j += 1

        assert len(indices) == i, f'Length mismatch: {len(indices)} vs {i}'

    if i / len(seq) < 0.9:
        logger.error(
            f"Sequence alignment mismatch: {seq} vs {chain_seq}"
        )
        return []
    assert len(indices) == len(
        seq), f'Length mismatch: {len(indices)} vs {len(seq)}'
    return indices


def label2bin(labels: torch.Tensor, metals: List[str]):
    assert labels.ndim == 1, f"labels should be 1D tensor, got {labels.ndim}D"
    assert labels.dtype == torch.int64, f"labels should be int64 tensor, got {labels.dtype}"

    out_tensor = torch.zeros(
        (len(labels), len(metals)), dtype=torch.long)
    for i, m in enumerate(metals):
        out_tensor[:, i] = labels == metal2token[m]

    return out_tensor


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: str, plm_name: str, ds_name: str, metal: str):
        lines = open(data_file).read().splitlines()
        self.data_list = []
        Path(
            f"cache/biolip/{plm_name}").mkdir(exist_ok=True, parents=True)
        if all(
            [
                Path(f"cache/biolip/{plm_name}/{pdbid[1:]}.pt").exists()
                for pdbid in lines[::3]
            ]
        ):
            logger.info(f"Loading {ds_name} from cache")
            self.data_list = [lines[i][1:] for i in range(0, len(lines), 3)]
        else:
            plm_func = get_model(plm_name, "cuda")
            for i in tqdm(range(0, len(lines), 3), desc=f"Processing {ds_name}"):
                if Path(f"cache/biolip/{plm_name}/{lines[i][1:]}.pt").exists():
                    continue
                seqid, seq, label = lines[i][1:], lines[i + 1], lines[i + 2]
                if plm_name in ['esmc-6b-2024-12', 'esmc_600m', 'esmc_300m']:
                    emb = plm_func([seq]).half()
                else:
                    emb = plm_func([seq])[..., 1].half()
                label = torch.tensor(list(map(int, label.split(','))))
                torch.save(
                    (seqid, emb, label), f"cache/biolip/{plm_name}/{seqid}.pt")

                self.data_list.append(seqid)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class SeqDataModule(L.LightningDataModule):
    def __init__(self, args: Args) -> None:
        super().__init__()
        self.args = args
        metal = args.metal
        self.train_set = SeqDataset(args.train_file, args.plm, "train", metal)
        self.val_set = SeqDataset(args.val_file, args.plm, "val", metal)
        self.test_set = SeqDataset(args.test_file, args.plm, "test", metal)

    def seq_collate_fn(self, plm_name, batch):
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=partial(self.seq_collate_fn, self.args.plm),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=partial(self.seq_collate_fn, self.args.plm),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=partial(self.seq_collate_fn, self.args.plm),
        )


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: str, metal: str):
        self.data_list = []
        seqids = list(set([l.split(',')[0]
                      for l in open(data_file).read().splitlines()]))
        for seqid in tqdm(seqids, desc=f"Loading {metal} probes"):
            possible_sites, offset, kskp = torch.load(
                f"cache/probe/{metal}/{seqid}.pt", map_location='cpu')
            self.data_list.extend(
                zip([seqid] * len(offset), possible_sites.float(), offset.float(), kskp.float()))
        print(f"Loaded {len(self.data_list)} entries")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class BalancedSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size, pos_ratio=0.25):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio

        self.num_pos_per_batch = round(batch_size * pos_ratio)
        self.num_neg_per_batch = batch_size - self.num_pos_per_batch
        self.num_batches = min(
            len(pos_indices) // self.num_pos_per_batch,
            len(neg_indices) // self.num_neg_per_batch,
        )

    def __iter__(self):
        pos_pool = random.sample(
            self.pos_indices, self.num_batches * self.num_pos_per_batch)
        neg_pool = random.sample(
            self.neg_indices, self.num_batches * self.num_neg_per_batch)
        for i in range(self.num_batches):
            pos = pos_pool[i *
                           self.num_pos_per_batch:(i + 1) * self.num_pos_per_batch]
            neg = neg_pool[i *
                           self.num_neg_per_batch:(i + 1) * self.num_neg_per_batch]
            yield pos + neg

    def __len__(self):
        return self.num_batches


class ProbeDataModule(L.LightningDataModule):
    def __init__(self, args: Args) -> None:
        super().__init__()
        self.args = args
        self.train_set = ProbeDataset(
            f'data/biolip/by_metal/{args.metal}_train.txt', metal=args.metal)
        self.val_set = ProbeDataset(
            f'data/biolip/by_metal/{args.metal}_val.txt', metal=args.metal)
        self.test_set = ProbeDataset(
            f'data/biolip/by_metal/{args.metal}_test.txt', metal=args.metal)

        # if self.args.sample_ratio > 0:
        #     # compute positive and negative samples of training set
        #     self.pos_indices, self.neg_indices = [], []
        #     for i, (seqid, site, offset, kskp) in enumerate(self.train_set):
        #         # if torch.norm(offset) <= pos_thr:
        #         if torch.norm(offset) <= self.args.max_offset:
        #             self.pos_indices.append(i)
        #         elif torch.norm(offset) > self.args.max_offset:
        #             self.neg_indices.append(i)
        #         # 3 ~ 5 is ignored
        #     self.sampler = BalancedSampler(
        #         self.pos_indices,
        #         self.neg_indices,
        #         self.args.batch_size,
        #         pos_ratio=self.args.sample_ratio,
        #     )

    def collate_fn(self, batch):
        return batch

    def train_dataloader(self):
        # if self.args.sample_ratio > 0:
        #     return DataLoader(
        #         self.train_set,
        #         batch_sampler=self.sampler,
        #         num_workers=8,
        #         collate_fn=self.collate_fn,
        #     )
        # else:
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collate_fn,
        )


class wwPDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: str):
        self.data_list = open(data_file).read().splitlines()
        for pdbid in tqdm(self.data_list, desc="Loading wwPDB dataset"):
            pdbid = pdbid.strip()
            if not Path(f"data/cifs/{pdbid}.cif").exists():
                if Path(f'/database/mmCIF/{pdbid[1:3]}/{pdbid}.cif.gz').exists():
                    # unzip the cif.gz file
                    with open(f"data/cifs/{pdbid}.cif", "w") as f:
                        f.write(gzip.open(
                            f'/database/mmCIF/{pdbid[1:3]}/{pdbid}.cif.gz', 'rt').read())
                else:
                    with open(f"data/cifs/{pdbid}.cif", "w") as f:
                        f.write(requests.get(
                            f'https://files.rcsb.org/download/{pdbid.upper()}.cif'
                        ).text)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class wwPDB(L.LightningDataModule):
    def __init__(self, args: Args) -> None:
        super().__init__()
        self.args = args
        self.train_set = wwPDBDataset(args.train_file)
        self.val_set = wwPDBDataset(args.val_file)
        self.atoms = []

    def pdb_collate_fn(self, batch):
        data = []
        for pdbid in batch:
            if pdbid in self.atoms:
                data.append((pdbid, None))
            else:
                try:
                    atoms = get_structure(CIFFile.read(
                        f"data/cifs/{pdbid}.cif"))[0]
                except Exception as e:
                    continue
                atoms = atoms[
                    np.isin(atoms.element, list(elem2token.keys()))]
                data.append((pdbid, atoms))
                self.atoms.append(pdbid)
        return data

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.pdb_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.pdb_collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()
