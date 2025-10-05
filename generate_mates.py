import argparse
from pathlib import Path
import pymol
import tempfile
from biotite.structure.io.pdbx import CIFFile, get_structure
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mates for a given input file.")
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="Input file path.")
    parser.add_argument("-o", "--output", type=str,
                        required=True, help="Output file path.")
    args = parser.parse_args()

    tmp_folder = tempfile.mkdtemp()

    pdb_stem = Path(args.input).stem

    pymol.cmd.reinitialize()
    pymol.cmd.load(args.input, pdb_stem)
    pymol.cmd.symexp('sym', pdb_stem, 'all', 4)
    for obj in pymol.cmd.get_names('objects'):
        pymol.cmd.save(f'{tmp_folder}/{obj}.cif', obj)

    ciffiles = list(Path(tmp_folder).glob('*.cif'))
    if len(ciffiles) == 0:
        print(f"No CIF files generated in {tmp_folder}.")

    merged = []
    for ciffile in ciffiles:
        structure = get_structure(CIFFile.read(ciffile))[0]
        if ciffile.stem != pdb_stem:
            structure.chain_id = np.char.add(ciffile.stem, structure.chain_id)

        merged.append(structure)

    merged = struc.array(np.concatenate(merged))
    strucio.save_structure(Path(args.output, f'{pdb_stem}_mated.cif'), merged)
