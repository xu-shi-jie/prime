""" `prime` command line entry point.

Sub-commands are dispatched lazily so that `prime --help` and `prime info`
stay fast and do not import torch.
"""
import sys

from . import __version__, paths

USAGE = f"""PRIME {__version__} - identification of metal-binding sites in proteins.

Usage:
  prime predict --pdb FILE[:CHAIN] ... [--metal ZN] [--outdir DIR] [options]
  prime predict-all -i FILE[:CHAIN] [-o DIR] [-m METAL ...]
  prime download [METAL ...]        Pre-fetch model checkpoints (default: all)
  prime info                        Show paths, cached checkpoints and metals

Metals: {', '.join(paths.METALS)}

`prime predict --help` lists all prediction options.
"""


def _download(metals):
    metals = [m.upper() for m in metals] or paths.METALS
    for metal in metals:
        print(f"{metal}: {paths.seq_ckpt(metal)}")
        print(f"{metal}: {paths.probe_ckpt(metal)}")
    return 0


def _info():
    print(f"PRIME {__version__}")
    print(f"package      {__file__.rsplit('/', 1)[0]}")
    print(f"assets       {paths.ASSETS}")
    print(f"checkpoints  {paths.HF_REPO} (Hugging Face Hub)")
    print(f"PRIME_HOME   {paths.home()}")
    print("metals       " + ", ".join(
        f"{m}{'*' if paths._find_local(f'weights/{paths.PROBE_CKPTS[m]}') else ''}"
        for m in paths.METALS))
    print("             (* = structure model available locally)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "--help"

    if cmd in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if cmd in ("-V", "--version", "version"):
        print(__version__)
        return 0

    sys.argv = [f"prime {cmd}"] + argv[1:]
    if cmd == "predict":
        from .predict import main as run
        return run()
    if cmd == "predict-all":
        from .predict_all import main as run
        return run()
    if cmd == "download":
        return _download(argv[1:])
    if cmd == "info":
        return _info()

    print(f"prime: unknown command {cmd!r}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
