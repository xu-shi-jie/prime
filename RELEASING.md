# Releasing PRIME

PRIME is distributed as the Python package **`prime-metal`** (imported as `prime`).
The package contains code and small assets only; the ~6.5 GB of model
checkpoints are hosted on the Hugging Face Hub and downloaded on first use.

## 0. Once: upload the checkpoints

```bash
pip install huggingface_hub
huggingface-cli login
python scripts/upload_hf_weights.py --dry_run   # check the file list first
python scripts/upload_hf_weights.py
```

This creates `https://huggingface.co/xushijie/prime` with the same layout as
the repository (`weights/...`, `checkpoints/seq_predictors/...`). Users can
point PRIME at a different repository with `PRIME_HF_REPO`, or at a local copy
with `PRIME_HOME`. **Never rename checkpoint files** — the file names carry the
metal type and the number of ResNet layers.

## 1. PyPI

```bash
pip install build twine
rm -rf dist/ && python -m build          # builds the sdist and the wheel
twine check dist/*
twine upload dist/*                      # needs a PyPI API token
```

For a new version, bump it in **both** `pyproject.toml` and
`prime/__init__.py`, then tag the release (`git tag v1.0.0 && git push --tags`).

## 2. Bioconda

Bioconda builds from the PyPI sdist, so release to PyPI first.

1. Fork and clone [bioconda/bioconda-recipes](https://github.com/bioconda/bioconda-recipes).
2. Copy the recipe and update the checksum:
   ```bash
   sha256sum dist/prime_metal-1.0.0.tar.gz      # paste into meta.yaml
   mkdir -p bioconda-recipes/recipes/prime-metal
   cp conda-recipe/meta.yaml bioconda-recipes/recipes/prime-metal/
   ```
3. Lint locally (optional but saves a CI round-trip):
   ```bash
   conda create -n bioconda -c conda-forge -c bioconda bioconda-utils
   conda activate bioconda
   bioconda-utils lint recipes config.yml --packages prime-metal
   bioconda-utils build recipes config.yml --packages prime-metal
   ```
4. Open a pull request from a branch of your fork. CI builds the recipe; when
   it is green a maintainer merges it and the package is published to the
   `bioconda` channel within about an hour.
5. For later versions, bump `version` and `sha256` in `meta.yaml` and reset
   `build: number` to `0`. (Bioconda's autobump bot usually opens that PR for
   you once the recipe exists.)

Notes:

* The recipe is `noarch: python` — one build works on every platform.
* `pytorch` comes from conda-forge and defaults to the CPU build. PRIME needs a
  CUDA GPU, so document `conda install -c conda-forge pytorch=*=*cuda*` for
  users, or let them install PyTorch themselves.
* Bioconda requires the source to be a versioned tarball with a checksum; do
  not point the recipe at a git branch.
