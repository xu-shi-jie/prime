import shutil
import numpy as np
from biotite.structure.io.pdb import PDBFile, get_structure
import biotite.structure.io as strucio
import biotite.structure as struc

from tqdm import tqdm
from pathlib import Path
import pymol
from pymol import cmd
# fmt: off
import sys
sys.path.append('.')
from models.dataset import considered_metals
# fmt: on
if __name__ == '__main__':
    pymol.finish_launching(['pymol', '-qc'])
    pymol.cmd.feedback("disable", "all", "everything")
    (out := Path('data/fixed_cifs')).mkdir(exist_ok=True, parents=True)
    pdbids = []
    for metal in tqdm(considered_metals, desc='Processing metals', dynamic_ncols=True):
        test_file = f'data/biolip/by_metal/{metal}_test.txt'
        val_file = f'data/biolip/by_metal/{metal}_val.txt'
        train_file = f'data/biolip/by_metal/{metal}_train.txt'

        lines = open(test_file).read().splitlines() + \
            open(val_file).read().splitlines() + \
            open(train_file).read().splitlines()
        pdbids.extend([l[:4] for l in lines])

    pdbids = sorted(set(pdbids))
    avail_pdbids = []
    for pdbid in pdbids:
        if (out / f'{pdbid}/{pdbid}.cif').exists():
            continue
        avail_pdbids.append(pdbid)

    for pdbid in tqdm(avail_pdbids, desc='Processing PDB IDs', dynamic_ncols=True):
        shutil.rmtree(out/pdbid, ignore_errors=True)
        (out/pdbid).mkdir(exist_ok=True, parents=True)
        pymol.cmd.reinitialize()
        pymol.cmd.load(f'data/cifs/{pdbid}.cif', pdbid)
        pymol.cmd.symexp('sym', pdbid, 'all', 4)
        for obj in pymol.cmd.get_names('objects'):
            pymol.cmd.save(out/pdbid/f'{obj}.cif', obj)

    pymol.cmd.quit()

    # merge all atoms
    for pdbid in (pbar := tqdm(pdbids, dynamic_ncols=True)):
        outdir = out/pdbid
        if not outdir.exists():
            continue

        pdbfiles = list(outdir.glob('*.cif'))
        if len(pdbfiles) == 0:
            continue

        if Path(outdir, f'{pdbid}.cif').exists():
            continue

        merged = []

        filesize = sum([pdbfile.stat().st_size for pdbfile in pdbfiles])
        pbar.set_description(f'Merging {pdbid} ({filesize / 1e6:.2f} MB)')
        for pdbfile in tqdm(pdbfiles, desc='Loading PDB files', leave=False):
            structure = get_structure(PDBFile.read(pdbfile))[0]
            if pdbfile.stem != pdbid:
                structure.chain_id = np.char.add(
                    pdbfile.stem, structure.chain_id)
            # merged.extend(structure)
            merged.append(structure)
        # merged = struc.concatenate(merged)
        merged = struc.array(np.concatenate(merged))
        # merged = struc.array(merged)
        strucio.save_structure(Path(outdir, f'{pdbid}.cif'), merged)
