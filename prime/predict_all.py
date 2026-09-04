import argparse
import subprocess
import sys

from .paths import METALS

def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Predict binding sites of all metals supported by PRIME.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file path, as file.cif[:chain]")
    parser.add_argument("-o", "--outdir", type=str, default='.', help="Output directory")
    parser.add_argument("-m", "--metals", nargs='+', default=METALS, help="Metal types to predict")
    args, extra = parser.parse_known_args()
    # fmt: on

    for metal in args.metals:
        subprocess.run([
            sys.executable, '-m', 'prime.predict',
            '--pdb', args.input,
            '--outdir', args.outdir,
            '--metal', metal,
            '--overwrite',
        ] + extra)

    print("All predictions completed.")


if __name__ == "__main__":
    main()
