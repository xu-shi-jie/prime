""" Upload the PRIME checkpoints to the Hugging Face Hub.

The pip / conda package ships code only; the checkpoints (~6.5 GB) are pulled
from the Hub on first use by `prime.paths`. Run this once from the repository
root after `huggingface-cli login`:

    python scripts/upload_hf_weights.py                    # upload everything
    python scripts/upload_hf_weights.py --metals ZN CA     # only some metals

The layout on the Hub must mirror the repository, because the checkpoint file
names carry metadata used at prediction time:

    weights/probe_<METAL>_resnet152_....ckpt
    checkpoints/seq_predictors/<METAL>.ckpt
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

# run from a checkout without installing the package first
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prime.paths import HF_REPO, METALS, PROBE_CKPTS  # noqa: E402


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=HF_REPO, help="Target Hugging Face model repository")
    parser.add_argument("--metals", nargs="+", default=METALS, help="Metal types to upload")
    parser.add_argument("--private", action="store_true", help="Create the repository as private")
    parser.add_argument("--dry_run", action="store_true", help="Only list what would be uploaded")
    args = parser.parse_args()
    # fmt: on

    files = []
    for metal in [m.upper() for m in args.metals]:
        files.append(Path("weights") / PROBE_CKPTS[metal])
        files.append(Path("checkpoints/seq_predictors") / f"{metal}.ckpt")

    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(
            "Missing checkpoints (run from the repository root, with git-lfs "
            "files pulled):\n  " + "\n  ".join(map(str, missing)))

    total = sum(f.stat().st_size for f in files) / 1024 ** 3
    print(f"{len(files)} file(s), {total:.1f} GB -> {args.repo}")
    if args.dry_run:
        for f in files:
            print(f"  {f}")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="model",
                    private=args.private, exist_ok=True)
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f}")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.as_posix(),
            repo_id=args.repo,
            repo_type="model",
        )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
