import os
from pathlib import Path
import posixpath
import subprocess
import argparse
import re
from loguru import logger
import pandas as pd
import paramiko
from tqdm import tqdm
import numpy as np


def parse_group(ckpt):
    match = re.search(
        r'probe_(?P<metal>\w+)_'
        r'(?P<cnn_layers>resnet\d+)_'
        r'pretrained=(?P<pretrained>True|False)_'
        r'(?P<loss_type>.+?)_epoch=(?P<epoch>\d+)',
        ckpt.stem
    )
    if match:
        return {
            'file': ckpt,
            'metal': match.group('metal'),
            'cnn_layers': match.group('cnn_layers'),
            'pretrained': match.group('pretrained') == 'True',
            'loss_type': match.group('loss_type'),
            'epoch': int(match.group('epoch')),
        }
    else:
        raise ValueError(f"Could not parse checkpoint name: {ckpt.stem}")


if __name__ == '__main__':
    out_csv = 'benchmark/scripts/prime_persistent/hard_mining.csv'
    ckpts = [Path(
        'checkpoints/tune_probe_predictor/probe_ZN_resnet152_pretrained=True_hard_mining_epoch=26_val_auc=0.9910.ckpt')]
    ckpts.extend(
        list(Path('checkpoints/tune_metal_predictor.bak').glob('*.ckpt')))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    outputs = []

    for ckpt in ckpts:
        logger.success(f'Processing checkpoint: {ckpt}')
        info = parse_group(ckpt)

        subprocess.run([
            'python',  'benchmark/scripts/prime/1.run_predictor.py',
            '--metal', info['metal'], '--ckpt', str(ckpt),
        ])
        subprocess.run([
            'python', 'benchmark/scripts/prime/2.compute_metrics.py',
            '--metal', info['metal'],
        ])
        result = pd.read_csv(
            f'benchmark/results/prime/metrics_{info["metal"]}.csv')
        test_row = {
            'file': ckpt.name,
            'metal': info['metal'],
            'cnn_layers': info['cnn_layers'],
            'pretrained': info['pretrained'],
            'loss_type': info['loss_type'],
            'epoch': info['epoch'],
            'test_TP': result['TP'].values[0],
            'test_FP': result['FP'].values[0],
            'test_FN': result['FN'].values[0],
            'test_F1': result['F1'].values[0],
            'test_RMSE': result['RMSE'].values[0],
            'dataset': 'test',
        }
        outputs.append(test_row)

    outpus = pd.DataFrame(outputs)
    outpus.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
