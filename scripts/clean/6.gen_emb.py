# Date: 2025-06-07
# Author: Shijie Xu
# Description: Generate embeddings for sequences in data/biolip/seq, not needed because dataset contains embeddings pre-computed
import torch
from pathlib import Path
from tqdm import tqdm
# fmt: off
import sys
sys.path.append('.')
from models.plm import get_model
# fmt: on

if __name__ == "__main__":
    lines = open('data/biolip/seq/seq_train.fasta').read().splitlines() +\
        open('data/biolip/seq/seq_val.fasta').read().splitlines() +\
        open('data/biolip/seq/seq_test.fasta').read().splitlines()
    seqs = {}
    for i in range(0, len(lines), 3):
        seqid = lines[i][1:]
        seq = lines[i + 1]
        label = list(map(int, lines[i + 2].split(',')))
        seqs[seqid] = (seq, label)

    for plm in ['esm2_t33_650M_UR50D', 'prot_t5_xl_half_uniref50-enc', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D',]:
        Path(f"cache/biolip/{plm}").mkdir(exist_ok=True, parents=True)
        plm_func = get_model(plm, 'cuda')
        for seqid, (seq, label) in tqdm(seqs.items(), desc=f"Generating embeddings for {plm}"):
            out_file = f"cache/biolip/{plm}/{seqid}.pt"
            if Path(out_file).exists():
                continue
            emb = plm_func([seq])[..., 1].half()
            label = torch.tensor(label, dtype=torch.int64)
            torch.save((seqid, emb, label), out_file)
