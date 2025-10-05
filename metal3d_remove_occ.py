import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove atoms with occupancy < 0.5 from PDB file.")
    parser.add_argument("-i", "--input_file", type=str,
                        required=True, help="Input PDB file.")
    parser.add_argument("-o", "--output_file", type=str,
                        required=True, help="Output PDB file.")
    args = parser.parse_args()

    with open(args.input_file) as f_in, open(args.output_file, "w") as f_out:
        for line in f_in:
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                occupancy = float(line[54:60])
                if occupancy >= 0.5:
                    f_out.write(line)
            else:
                f_out.write(line)
