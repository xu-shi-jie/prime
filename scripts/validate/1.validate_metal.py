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
    # Example filename: probe_CA_resnet152_pretrained=True_fp_penalty_epoch=0.ckpt
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
    # fmt: off
    parser = argparse.ArgumentParser(description='Run seq_predictor.py with specific checkpoints.')
    parser.add_argument('--out_csv', type=str, default='benchmark/scripts/prime_persistent/all_metals.csv')
    args = parser.parse_args()
    # fmt: on

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    hostname, port, username, private_key = \
        open('hpc_config').read().splitlines()
    private_key = paramiko.RSAKey.from_private_key_file(private_key)
    ssh.connect(hostname, port, username, pkey=private_key)
    sftp = ssh.open_sftp()

    patience = 5
    folders = [
        'checkpoints/tune_metal_predictor/',
        'checkpoints/tune_neg_penalty_predictor/',
    ]
    remote_folders = [
        '/work/a30171/xushijie/moved_checkpoints/',
        '/work/a30171/xushijie/prime/checkpoints/tune_neg_penalty_predictor/'
    ]
    if not Path(args.out_csv).exists():
        # write empty csv file
        df = pd.DataFrame(columns=[
            'file', 'metal', 'cnn_layers', 'pretrained', 'loss_type', 'epoch',
            'TP', 'FP', 'FN', 'F1', 'RMSE', 'dataset',
        ])
        df.to_csv(args.out_csv, index=False)

    df = pd.read_csv(args.out_csv)
    ckpts, remote_ckpts = [], []
    for folder in folders:
        ckpts.extend(list(Path(folder).glob('*.ckpt')))
    print(f"Found {len(ckpts)} local checkpoints.")
    for folder in remote_folders:
        for entry in sftp.listdir_attr(folder):
            filename = entry.filename
            if filename.endswith('.ckpt'):
                full_path = posixpath.join(folder, filename)
                remote_ckpts.append(Path(full_path))
    print(f"Found {len(remote_ckpts)} remote checkpoints.")
    checked_ckpts = {}
    for _, row in df.iterrows():
        if row['dataset'] == 'val':
            checked_ckpts[row['file']] = row['F1'] - row['RMSE']

    print(
        f"Found {len(ckpts)+len(remote_ckpts)} checkpoints, {len(checked_ckpts)} already checked.")

    ckpt_info = []
    for ckpt in ckpts:
        info = parse_group(ckpt)
        info['remote'] = False
        ckpt_info.append(info)
    for ckpt in remote_ckpts:
        info = parse_group(ckpt)
        info['remote'] = True
        ckpt_info.append(info)

    print(f'All metals found: {set(info["metal"] for info in ckpt_info)}')
    input("Press Enter to continue...")
    # sort by metal then epoch
    ckpt_info.sort(key=lambda x: (x['metal'], x['epoch']))

    waiting = {}  # waiting for each metal
    best_val = {}  # best validation results for each metal
    for i, info in enumerate(ckpt_info):
        logger.success(
            f'Processing checkpoint {i+1}/{len(ckpt_info)}: {info["file"]}')
        metal, ckpt = info['metal'], info['file']
        if waiting.get(metal, 0) > patience:
            print(f"Skipping {ckpt}, waiting for {metal} exceeded patience.")
            continue

        file_name = ckpt.name
        if file_name in checked_ckpts:
            print(f"Skipping {ckpt}, already checked.")
            if checked_ckpts[file_name] > best_val.get(metal, (None, float('-inf')))[1]:
                best_val[metal] = (file_name, checked_ckpts[file_name])
                waiting[metal] = 0
            elif np.isnan(checked_ckpts[file_name]):
                print(f"NA encountered for {metal} in {ckpt.name}.")
                waiting[metal] = 0
            else:
                waiting[metal] = waiting.get(metal, 0) + 1

            continue
        print(f"Processing {ckpt} for metal {metal}...")

        if info['remote']:
            local_path = ckpt.name
            print(f"Downloading remote checkpoint {ckpt} to {local_path}...")
            sftp.get(str(ckpt), local_path)
            ckpt = Path(local_path)

        subprocess.run([
            'python', 'benchmark/scripts/prime/1.run_predictor.py',
            '--ckpt', str(ckpt), '--metal', metal, '--dataset', 'test',
        ])

        NA_VAL_OR_TEST = False
        try:
            subprocess.run([
                'python', 'benchmark/scripts/prime/2.compute_metrics.py',
                '--metal', metal, '--dataset', 'test',
            ])
            result = pd.read_csv(
                f'benchmark/results/prime/metrics_{metal}.csv')
            test_row = {
                'file': ckpt.name,
                'metal': metal,
                'cnn_layers': info['cnn_layers'],
                'pretrained': info['pretrained'],
                'loss_type': info['loss_type'],
                'epoch': info['epoch'],
                'TP': result['TP'].values[0],
                'FP': result['FP'].values[0],
                'FN': result['FN'].values[0],
                'F1': result['F1'].values[0],
                'RMSE': result['RMSE'].values[0],
                'dataset': 'val',
            }
        except:
            test_row = {
                'file': ckpt.name,
                'metal': metal,
                'cnn_layers': info['cnn_layers'],
                'pretrained': info['pretrained'],
                'loss_type': info['loss_type'],
                'epoch': info['epoch'],
                'TP': float('nan'),
                'FP': float('nan'),
                'FN': float('nan'),
                'F1': float('nan'),
                'RMSE': float('nan'),
                'dataset': 'val',
            }
            NA_VAL_OR_TEST = True

        subprocess.run([
            'python', 'benchmark/scripts/prime/1.run_predictor.py',
            '--ckpt', str(ckpt), '--metal', metal, '--dataset', 'val',
        ])
        try:
            subprocess.run([
                'python', 'benchmark/scripts/prime/2.compute_metrics.py',
                '--metal', metal, '--dataset', 'val',
            ])
            result = pd.read_csv(
                f'benchmark/results/prime/metrics_{metal}.csv')
            val_row = {
                'file': ckpt.name,
                'metal': metal,
                'cnn_layers': info['cnn_layers'],
                'pretrained': info['pretrained'],
                'loss_type': info['loss_type'],
                'epoch': info['epoch'],
                'TP': result['TP'].values[0],
                'FP': result['FP'].values[0],
                'FN': result['FN'].values[0],
                'F1': result['F1'].values[0],
                'RMSE': result['RMSE'].values[0],
                'dataset': 'test',
            }
        except:
            val_row = {
                'file': ckpt.name,
                'metal': metal,
                'cnn_layers': info['cnn_layers'],
                'pretrained': info['pretrained'],
                'loss_type': info['loss_type'],
                'epoch': info['epoch'],
                'TP': float('nan'),
                'FP': float('nan'),
                'FN': float('nan'),
                'F1': float('nan'),
                'RMSE': float('nan'),
                'dataset': 'test',
            }
            NA_VAL_OR_TEST = True

        df = pd.concat([
            df,
            pd.DataFrame([test_row]),
            pd.DataFrame([val_row])
        ], ignore_index=True)

        df.to_csv(args.out_csv, index=False)

        if NA_VAL_OR_TEST:
            logger.warning(f"NA encountered for {metal} in {ckpt.name}. ")
            waiting[metal] = 0
            continue
        else:
            metric = val_row['F1'] - val_row['RMSE']

        best_metric = best_val.get(metal, (None, float('-inf')))[1]

        if metric > best_metric:
            best_val[metal] = (ckpt.name, metric)
            waiting[metal] = 0
            print(
                f"New best for {metal}: {ckpt.name} with metric {metric:.4f}")
        else:
            waiting[metal] = waiting.get(metal, 0) + 1
            print(
                f"Waiting for {metal}, current waiting count: {waiting[metal]}")
            if waiting[metal] > patience:
                print(
                    f"Patience exceeded for {metal}, skipping further checks.")
                continue

        # delete the local checkpoint file if it was downloaded
        if info['remote']:
            print(f"Deleting local checkpoint {ckpt}...")
            ckpt.unlink()

    os.system('rm probe_*.ckpt')
    print("All checkpoints processed.")
