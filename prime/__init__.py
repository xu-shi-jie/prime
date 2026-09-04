""" PRIME: Probe-based identification of metal-binding sites using deep learning representations.

    >>> from prime.predict import main   # command line entry point
"""
from .paths import METALS, PROBE_CKPTS, config, probe_ckpt, seq_ckpt, stats_csv

__version__ = "1.0.1"
__all__ = ["METALS", "PROBE_CKPTS", "config",
           "probe_ckpt", "seq_ckpt", "stats_csv", "__version__"]
