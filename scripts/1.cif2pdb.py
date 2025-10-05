from Bio.PDB import MMCIFParser, PDBIO
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


def batch_fix(pdbid):
    if Path(f'data/pdbs/{pdbid}.pdb').exists():
        print(f"{pdbid} already exists, skipping.")
        return
    # Load CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", f'data/cifs/{pdbid}.cif')
    try:
        # Save as PDB
        io = PDBIO()
        io.set_structure(structure)
        io.save(f"data/pdbs/{pdbid}.pdb")
    except Exception as e:
        print(f"Error processing {pdbid}: {e}")
        return


if __name__ == '__main__':
    lines = open('data/biolip/seq/seq_test.fasta').read().splitlines()
    pdbids = list(set([l[1:5] for l in lines if l.startswith('>')]))
    Path('data/pdbs').mkdir(exist_ok=True, parents=True)
    with mp.Pool(16) as pool:
        list(tqdm(
            pool.imap(batch_fix, pdbids),
            total=len(pdbids),
            desc="Converting CIF to PDB"
        ))
    print("All PDB files have been converted from CIF to PDB format.")
