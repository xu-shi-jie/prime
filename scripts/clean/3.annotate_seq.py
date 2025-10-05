import os
from pathlib import Path
from Bio.SeqIO import parse
from tqdm import tqdm
# fmt: off
import sys
sys.path.append(".")
from models.dataset import read_biolip, metals, metal2token
# fmt: on

if __name__ == "__main__":
    # merge all train, val, test files
    Path("data/biolip/seq/seq_train.fasta").unlink(missing_ok=True)
    Path("data/biolip/seq/seq_val.fasta").unlink(missing_ok=True)
    Path("data/biolip/seq/seq_test.fasta").unlink(missing_ok=True)
    os.system('cat data/biolip/seq/*_train.fasta > data/biolip/seq/seq_train.fasta')
    os.system('cat data/biolip/seq/*_val.fasta > data/biolip/seq/seq_val.fasta')
    os.system('cat data/biolip/seq/*_test.fasta > data/biolip/seq/seq_test.fasta')

    data_dict = {}
    for f in Path("data/biolip/seq").glob("seq_*.fasta"):
        fasta = parse(f, "fasta")
        for r in tqdm(fasta, desc="Converting to Dict"):
            seq = str(r.seq)
            data_dict[r.id] = str(seq), [-1] * len(seq)

    biolip = read_biolip("data/biolip/BioLiP.txt")
    for i, row in tqdm(biolip.iterrows(), desc="Annotating"):
        pdb_id = row["PDB ID"]
        chain_id = row["Receptor chain"]
        lig_name = row["Ligand ID"]
        if lig_name not in metals:
            continue
        seq, labels = data_dict.get(f"{pdb_id}{chain_id}", ("", []))
        if seq == "":
            continue

        binding_res = row["Binding site residues (Re-numbered)"]
        for res in binding_res.split():
            res_name = res[0]
            res_id = int(res[1:])
            labels[res_id - 1] = metal2token[lig_name]
        data_dict[f"{pdb_id}{chain_id}"] = seq, labels

    train_ids = [r.id for r in parse(
        "data/biolip/seq/seq_train.fasta", "fasta")]
    val_ids = [r.id for r in parse(
        "data/biolip/seq/seq_val.fasta", "fasta")]
    test_ids = [r.id for r in parse(
        "data/biolip/seq/seq_test.fasta", "fasta")]

    with open("data/biolip/seq/seq_train.fasta", "w") as f:
        for k, v in data_dict.items():
            if k in train_ids:
                f.write(f">{k}\n{v[0]}\n{','.join(map(str, v[1]))}\n")

    with open("data/biolip/seq/seq_val.fasta", "w") as f:
        for k, v in data_dict.items():
            if k in val_ids:
                f.write(f">{k}\n{v[0]}\n{','.join(map(str, v[1]))}\n")

    with open("data/biolip/seq/seq_test.fasta", "w") as f:
        for k, v in data_dict.items():
            if k in test_ids:
                f.write(f">{k}\n{v[0]}\n{','.join(map(str, v[1]))}\n")
