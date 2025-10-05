# Author: Shijie Xu
# Date: 2024-10-12
# This script will download the BioLiP dataset and protein sequences from the Zhang Lab website.

import subprocess
from pathlib import Path

if __name__ == '__main__':
    Path('data/biolip/BioLiP.txt').unlink(missing_ok=True)
    Path('data/biolip/protein.fasta').unlink(missing_ok=True)
    Path('data/biolip/BioLiP.txt.gz').unlink(missing_ok=True)
    Path('data/biolip/protein.fasta.gz').unlink(missing_ok=True)

    Path('data/biolip').mkdir(exist_ok=True, parents=True)
    # download biolip
    subprocess.run([
        'aria2c', '-x', '16', 'https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz', '-d', 'data/biolip'])
    # download protein sequences
    subprocess.run([
        'aria2c', '-x', '16', 'https://zhanggroup.org/BioLiP/data/protein.fasta.gz', '-d', 'data/biolip'])
    # unzip
    subprocess.run(['gunzip', 'data/biolip/BioLiP.txt.gz'])
    subprocess.run(['gunzip', 'data/biolip/protein.fasta.gz'])
