""" Backwards-compatible alias: the `models` package now lives in `prime.models`.

Keeps `from models.xxx import yyy` working in the training scripts and notebooks
of this repository. New code should import `prime.models.xxx` instead.
"""
import sys

from prime import models as _models

sys.modules[__name__] = _models
