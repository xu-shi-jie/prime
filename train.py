import argparse
import warnings
import sys

from loguru import logger

from models.utils import Config
import torch

torch.set_float32_matmul_precision("medium")

logger.configure(handlers=[{"sink": sys.stdout, "level": "ERROR"}])
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # from models.training import pretrain
    # args = Config("configs/pretrain.yaml")
    # pretrain(args)

    # from models.training import train_seq_predictor
    # args = Config("configs/train_seq_predictor.yaml")
    # train_seq_predictor(args)

    from models.training import train_probe_predictor
    args = Config("configs/train_structure_predictor.yaml")
    train_probe_predictor(args)
