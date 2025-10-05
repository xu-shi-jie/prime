import subprocess
from models.dataset import considered_metals
from pathlib import Path
import argparse

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Predict all metals in the dataset.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input file path")
    parser.add_argument("-o", "--outdir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    # fmt: on

    checkpoints = Path('weights').glob('*.ckpt')
    for ckpt in checkpoints:
        subprocess.run([
            'python', 'predict.py',
            '--pdb', args.input,
            '--outdir', args.outdir,
            '--ckpt', str(ckpt),
            '--overwrite',
        ])

    print("All predictions completed.")
