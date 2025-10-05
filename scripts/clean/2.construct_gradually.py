# fmt: off
import multiprocessing as mp
import pickle
import requests
from math import ceil, floor
import random
import shutil
from Bio.SeqIO import parse
import shlex
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


def read_or_download(pdbid):
    Path('data/fasta').mkdir(exist_ok=True, parents=True)
    fasta_file = f'data/fasta/{pdbid}.fasta'
    if not Path(fasta_file).exists():
        with open(fasta_file, 'w') as f:
            f.write(requests.get(
                    f'https://www.rcsb.org/fasta/entry/{pdbid}').text)
    return [(r.id, str(r.seq)) for r in parse(fasta_file, 'fasta')]


def read_lm(file):
    lines = open(file).read().splitlines()
    return [(lines[i][1:], lines[i+1]) for i in range(0, len(lines), 3)]


def cluster(fasta_file):
    for cmd in [
        f"mmseqs easy-cluster {fasta_file} clusterRes tmp --min-seq-id 0.3 -c 0.8 --cov-mode 1",
    ]:
        subprocess.run(
            cmd, shell=True, check=True,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )


def search(fasta1, fasta2):
    cmd = f'mmseqs easy-search {fasta1} {fasta2} alnRes.m8 tmp --min-seq-id 0.3 -c 0.8 --cov-mode 1'
    subprocess.run(
        cmd, check=True, shell=True,
        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )


if __name__ == '__main__':
    metal3d_train_ZN = []
    pdbids = open(
        'benchmark/compared_methods/metal-site-prediction/data/train.txt').read().splitlines()
    with mp.Pool(mp.cpu_count()) as pool:
        metal3d_train_ZN = list(
            tqdm(
                pool.imap(read_or_download, pdbids), total=len(pdbids),
                desc='Downloading Metal3D ZN training sequences', dynamic_ncols=True, leave=False))
    metal3d_train_ZN = [
        item for sublist in metal3d_train_ZN for item in sublist]
    lm_train_ZN = read_lm(
        'benchmark/compared_methods/LMetalSite/datasets/ZN_Train_1647.fa')
    lm_train_CA = read_lm(
        'benchmark/compared_methods/LMetalSite/datasets/CA_Train_1554.fa')
    lm_train_MG = read_lm(
        'benchmark/compared_methods/LMetalSite/datasets/MG_Train_1730.fa')
    lm_train_MN = read_lm(
        'benchmark/compared_methods/LMetalSite/datasets/MN_Train_547.fa')
    lm_train = lm_train_ZN + lm_train_CA + lm_train_MG + lm_train_MN
    # read m-ionic
    mionic_df = pd.read_csv(
        'benchmark/compared_methods/m-ionic/data/pos_data/LigID_pdbchain_partitions.csv')
    mionic_train_ids = mionic_df[
        mionic_df['partitions'] != 6]['pdbchain'].values
    mionic_data = pickle.load(
        open('benchmark/compared_methods/m-ionic/data/pos_data/multi_ion.pkl', 'rb'))

    biolip = read_biolip('data/biolip/BioLiP.txt')
    metals = considered_metals

    shutil.rmtree('data/biolip/seq', ignore_errors=True)
    Path('data/biolip/seq').mkdir(parents=True, exist_ok=True)
    shutil.rmtree('tmp', ignore_errors=True)
    Path('tmp').mkdir(parents=True, exist_ok=True)
    if Path('scripts/clean/invalid_pdbids.txt').exists():
        invalid_pdbids = set(
            open('scripts/clean/invalid_pdbids.txt').read().splitlines())
    else:
        invalid_pdbids = set()
    available_pdbids = set(
        open('data/biolip/available_pdbs.txt').read().splitlines())
    biolip = biolip[biolip['PDB ID'].isin(available_pdbids)]

    metal_counts = {}
    for metal in considered_metals:
        metal_counts[metal] = len(biolip[biolip['Ligand ID'] == metal])
    metals = sorted(metal_counts, key=metal_counts.get, reverse=True)
    print(
        f"Gradually constructing datasets for metals: {', '.join(metals[::-1])}")

    used_seqs = []
    for metal in metals[::-1]:  # process metals in reverse order
        print(f'{metal}: ', end='')
        metal_df = biolip[biolip['Ligand ID'] == metal]
        print(f" {len(metal_df)} raw,", end='')
        target_seqs = {}
        for i, row in metal_df.iterrows():
            seqid = row['PDB ID'] + row['Receptor chain']
            seq = row['Receptor sequence']
            if len(seq) > 2048:
                continue
            target_seqs[seqid] = seq
        target_seqs = list(target_seqs.items())
        print(f' {len(target_seqs)} unique,', end='')

        excluded_seqs = []

        if metal in ['ZN', 'CA', 'MG', 'MN', 'FE', 'CU', 'FE2']:  # remove test overlapping
            if metal == 'ZN':
                excluded_seqs = lm_train + metal3d_train_ZN
                print(
                    f' {len(lm_train)} LM, {len(metal3d_train_ZN)} Metal3D,', end='')
            mionic_train = []
            for seqid in mionic_train_ids:
                seqid_data = mionic_data[seqid]
                mionic_train.append((seqid, seqid_data['seq']))
            excluded_seqs += mionic_train
            print(f' {len(mionic_train)} m-ionic,', end='')

        print(f' {len(used_seqs)} used,', end='')
        excluded_seqs += used_seqs

        with open(f'tmp/{metal}_target.fasta', 'w') as f:
            for seqid, seq in target_seqs:
                f.write(f">{seqid}\n{seq}\n")
        with open(f'tmp/{metal}_excluded.fasta', 'w') as f:
            for seqid, seq in excluded_seqs:
                f.write(f">{seqid}\n{seq}\n")
        cluster(f'tmp/{metal}_target.fasta')
        # parse clusterRes_cluster.tsv
        cluster_df = pd.read_csv(
            'clusterRes_cluster.tsv', sep='\t', header=None, keep_default_na=False, na_values=None)
        clusters = {}
        for i, row in cluster_df.iterrows():
            repr_seq = row[0]
            clusters.setdefault(repr_seq, []).append(row[1])
        print(f' {len(clusters)} clusters,', end='')
        all_ids = set(clusters.keys())

        if len(excluded_seqs):
            search(f'tmp/{metal}_excluded.fasta', f'tmp/{metal}_target.fasta')
            # parse alnRes.m8
            # check if empty
            if os.stat('alnRes.m8').st_size == 0:
                searched_ids = set()
            else:
                aln_df = pd.read_csv(
                    'alnRes.m8', sep='\t', header=None, keep_default_na=False, na_values=None)
                aln_df.columns = [
                    'query', 'target', 'identity', 'alignment_length',
                    'mismatches', 'gap_opens', 'q_start', 'q_end',
                    't_start', 't_end', 'evalue', 'bit_score']
                aln_df = aln_df.groupby('target').agg({
                    'identity': 'max',
                }).reset_index()
                searched_ids = set(aln_df['target'].values)
            print(f' {len(all_ids & searched_ids)} intersected,', end='')
            possible_test_ids = all_ids - searched_ids
            train_ids = all_ids - possible_test_ids
            print(f' {len(searched_ids)} searched,', end='')
            train_ids = list(train_ids)
            possible_test_ids = list(possible_test_ids)
            print(f' {len(possible_test_ids)} possible test,', end='')

            random.seed(42)
            random.shuffle(possible_test_ids)
            n_total = len(all_ids)
            n_train, n_test = floor(0.8 * n_total), ceil(0.1 * n_total)
            n_val = n_total - n_train - n_test
            assert n_test <= len(possible_test_ids), \
                f"Not enough test ids for {metal}: {len(possible_test_ids)} available, {n_test} required"
            test_ids = possible_test_ids[:n_test]
            train_ids = possible_test_ids[n_test:] + train_ids
            random.shuffle(train_ids)
            val_ids = train_ids[:n_val]
            train_ids = train_ids[n_val:]
        else:
            all_ids = list(all_ids)
            random.seed(42)
            random.shuffle(all_ids)
            n_total = len(all_ids)
            n_train, n_test = floor(0.8 * n_total), ceil(0.1 * n_total)
            n_val = n_total - n_train - n_test
            train_ids = all_ids[:n_train]
            val_ids = all_ids[n_train:n_train + n_val]
            test_ids = all_ids[n_train + n_val:]

        print(
            f' {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test,', end='')
        with \
                open(f'data/biolip/seq/{metal}_train.fasta', 'w') as f1, \
                open(f'data/biolip/seq/{metal}_val.fasta', 'w') as f2, \
                open(f'data/biolip/seq/{metal}_test.fasta', 'w') as f3:
            for seqid, seq in target_seqs:
                if seqid in train_ids:
                    f1.write(f">{seqid}\n{seq}\n")
                    used_seqs.append((seqid, seq))
                elif seqid in val_ids:
                    f2.write(f">{seqid}\n{seq}\n")
                    used_seqs.append((seqid, seq))
                elif seqid in test_ids:
                    f3.write(f">{seqid}\n{seq}\n")
                    used_seqs.append((seqid, seq))
        print(
            f' {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test,')

    os.system('rm -rf tmp clusterRes* alnRes.m8')
