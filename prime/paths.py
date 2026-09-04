""" Locate the data files and model checkpoints shipped with / downloaded for PRIME.

Small assets (configs, metal-NOS-AA statistics) are bundled inside the wheel.
Model checkpoints are far too large for PyPI, so they are pulled on demand from
the Hugging Face Hub and cached locally.

Lookup order for a checkpoint:

1. an explicit path given by the user (``--ckpt``);
2. ``$PRIME_HOME/weights/...``   (set ``PRIME_HOME`` to reuse an existing copy);
3. ``./weights/...``            (running from a clone of the repository);
4. the Hugging Face Hub        (downloaded once, then cached).
"""
import os
from pathlib import Path

# The Hugging Face repository holding the checkpoints. Override with
# the PRIME_HF_REPO environment variable if you host your own mirror.
HF_REPO = os.environ.get("PRIME_HF_REPO", "xushijie/prime")
HF_REVISION = os.environ.get("PRIME_HF_REVISION", "main")

ASSETS = Path(__file__).parent / "assets"

# Structure ("probe") model file names, per metal type. The file names carry
# metadata used by the prediction program, so they must not be renamed.
PROBE_CKPTS = {
    "CA": "probe_CA_resnet152_pretrained=True_fp_penalty_epoch=18.ckpt",
    "CD": "probe_CD_resnet152_pretrained=True_hard_mining_epoch=12_val_auc=0.9711.ckpt",
    "CO": "probe_CO_resnet152_pretrained=True_hard_mining_epoch=48_val_auc=0.9896.ckpt",
    "CU": "probe_CU_resnet152_pretrained=True_hard_mining_epoch=3_val_auc=0.9577.ckpt",
    "CU1": "probe_CU1_resnet152_pretrained=True_hard_mining_epoch=11_val_auc=0.9694.ckpt",
    "FE": "probe_FE_resnet152_pretrained=True_fp_penalty_epoch=2.ckpt",
    "FE2": "probe_FE2_resnet152_pretrained=True_hard_mining_epoch=34_val_auc=0.9936.ckpt",
    "HG": "probe_HG_resnet152_pretrained=True_fp_penalty_epoch=47.ckpt",
    "K": "probe_K_resnet152_pretrained=True_hard_mining_epoch=5_val_auc=0.9755.ckpt",
    "MG": "probe_MG_resnet152_pretrained=True_hard_mining_epoch=31_val_auc=0.9608.ckpt",
    "MN": "probe_MN_resnet152_pretrained=True_hard_mining_epoch=40_val_auc=0.9935.ckpt",
    "NA": "probe_NA_resnet152_pretrained=True_hard_mining_epoch=21_val_auc=0.9885.ckpt",
    "NI": "probe_NI_resnet152_pretrained=True_hard_mining_epoch=33_val_auc=0.9911.ckpt",
    "ZN": "probe_ZN_resnet152_pretrained=True_hard_mining_epoch=26_val_auc=0.9910.ckpt",
}

# Metals PRIME can predict, i.e. those with both a sequence and a probe model.
METALS = sorted(PROBE_CKPTS)


def asset(*parts) -> str:
    """Path to a data file bundled with the package, e.g. asset('data', 'ions.txt')"""
    p = ASSETS.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"Missing bundled asset: {p}")
    return str(p)


def config(name: str) -> str:
    """Path to a bundled YAML config, e.g. config('seq_predictors/ZN.yaml')"""
    return asset("configs", *name.split("/"))


def stats_csv() -> str:
    """Path to the bundled metal-NOS-AA distance statistics"""
    return asset("data", "statistics_distance_residue.csv")


def home() -> Path:
    """Directory searched for locally available checkpoints"""
    return Path(os.environ.get("PRIME_HOME", ".")).expanduser()


def _find_local(relpath: str):
    """Return a local copy of `relpath` (e.g. 'weights/probe_ZN_....ckpt'), if any"""
    for root in (home(), Path(".")):
        p = root / relpath
        if p.exists():
            return str(p)
    return None


def _download(relpath: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download PRIME checkpoints. "
            "Install it with `pip install huggingface_hub`, or download the "
            f"checkpoints manually and point PRIME_HOME at their parent directory."
        )
    from loguru import logger
    logger.info(f"Downloading {relpath} from {HF_REPO} (this happens once) ......")
    return hf_hub_download(HF_REPO, relpath, revision=HF_REVISION)


def fetch(relpath: str) -> str:
    """Return a local path to `relpath`, downloading it from the Hub if needed"""
    return _find_local(relpath) or _download(relpath)


def probe_ckpt(metal: str) -> str:
    """Path to the structure (probe) model checkpoint of `metal`"""
    metal = metal.upper()
    if metal not in PROBE_CKPTS:
        raise ValueError(
            f"No structure model for metal {metal!r}. Available: {', '.join(METALS)}")
    return fetch(f"weights/{PROBE_CKPTS[metal]}")


def seq_ckpt(metal: str) -> str:
    """Path to the sequence model checkpoint of `metal`"""
    return fetch(f"checkpoints/seq_predictors/{metal.upper()}.ckpt")
