from argparse import Namespace as Args
import pickle
import random
from pathlib import Path
import shutil
import subprocess
import time
import os
import line_profiler
import rich
import torch
import e3nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from models.module import MLP, ConvFCLayers
import numpy as np
from math import ceil
from models.pdb import rand_rot
from timm.scheduler.cosine_lr import CosineLRScheduler
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import wandb
from models.dataset import SeqDataModule, ProbeDataModule, SeqDataset, label2bin, read_biolip, wwPDB, considered_metals,  abundants
from models.module import Header
from models.plm import EsmModelInfo
from models.resnet import generate_model, FocalLoss, batch_data
from models.utils import Config, is_wandb_running, kde_pytorch, log_kde, memory_usage_psutil, generate_id
from models.dataset import elem2token, res2token, _3to1, _1to3, metal2token, metals, token2metal, transition_metals, all_elem2token, all_token2elem, metal2elem, token2elem
from loguru import logger
from einops import rearrange
from biotite.structure.io.pdbx import CIFFile, get_structure
import time
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_sum
import pandas as pd


class SeqPredictor(L.LightningModule):
    def __init__(self, args: Args):
        super(SeqPredictor, self).__init__()
        self.args = args
        esminfo = EsmModelInfo(args.plm)
        self.model = Header(
            in_size=esminfo["dim"],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=2,
            net_type=args.rnn_type,
        )
        self.automatic_optimization = False

        self.cache = {}

    def forward(self, x):
        return self.model(x)

    def get_esm_feats(self, seqid):
        if seqid in self.cache:
            return self.cache[seqid]

        f = torch.load(
            f"cache/biolip/{self.args.plm}/{seqid}.pt",
            weights_only=True,
        )
        # memory might not be enough
        if len(self.cache) < 5000 or self.args.plm != 'esm2_t48_15B_UR50D':
            self.cache[seqid] = f
        return f

    def step(self, batch):
        batch = [self.get_esm_feats(seqid) for seqid in batch]
        seqids, seqs, labels = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in seqs]).to(self.device)

        seqs = nn.utils.rnn.pad_sequence(
            seqs, batch_first=True).to(self.device)

        preds = self.model(seqs, train_shared=(self.args.metal == 'all'))

        preds = nn.utils.rnn.unpad_sequence(
            preds, lengths, batch_first=True)
        preds = torch.cat(preds)
        labels = torch.cat(labels).to(self.device)

        # -1 -> 0, >=0 -> 1 # binary classification
        if self.args.metal == 'all':
            bin_labels = (labels >= 0).long()
        else:
            bin_labels = (labels == metal2token[self.args.metal]).long()

        if self.args.loss == 'ce':
            # option1: cross-entropy loss
            loss = F.cross_entropy(preds, bin_labels)
        # option2: focal loss
        elif self.args.loss == 'focal':
            loss = FocalLoss(gamma=2, reduction='mean')(
                preds, bin_labels)
        # option3: hard negative mining
        elif self.args.loss == 'hard_mining':
            pos_mask = bin_labels == 1
            neg_mask = ~pos_mask
            loss = F.cross_entropy(preds, bin_labels, reduction='none')
            pos_loss, neg_loss = loss[pos_mask], loss[neg_mask]
            # choose the top 20% of difficult negative samples
            num_hard_neg = int(len(neg_loss) * 0.2)
            hard_neg_loss, _ = torch.topk(neg_loss, num_hard_neg)
            loss = torch.cat([pos_loss, hard_neg_loss]).mean()
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.step(batch)
        self.log_dict({
            "train/loss": loss,
        },
            prog_bar=True)

        self.manual_backward(loss)
        self.optimizers().step()
        self.lr_schedulers().step(self.global_step)
        self.optimizers().zero_grad()

    def on_validation_epoch_start(self):
        self.probs, self.labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.step(batch)
        self.probs.append(preds)
        self.labels.append(labels)

    def on_validation_epoch_end(self):
        probs = torch.cat(self.probs)
        labels = torch.cat(self.labels)
        # compute auc, f1, mcc, pre, rec in each group
        metrics = {}
        for grp in labels.unique():
            if grp < 0:
                continue
            me = token2metal[grp.item()]
            mask = labels == grp
            rec = probs[mask].argmax(-1).float().mean()
            metrics.update({
                f"val_rec_{me}": rec,
            })
        if self.args.metal == 'all':
            labels = (labels >= 0).long()
        else:
            labels = (labels == metal2token[self.args.metal]).long()

        try:
            auc = roc_auc_score(labels.cpu(), probs.softmax(-1).cpu()[..., 1])
        except ValueError:
            auc = 0.0
        f1 = f1_score(labels.cpu(), probs.argmax(-1).cpu())
        mcc = matthews_corrcoef(labels.cpu(), probs.argmax(-1).cpu())
        pre = precision_score(labels.cpu(), probs.argmax(-1).cpu())
        rec = recall_score(labels.cpu(), probs.argmax(-1).cpu())
        metrics.update({
            "val_auc": auc,
            "val_f1": f1,
            "val_mcc": mcc,
            "val_pre": pre,
            "val_rec": rec,
        })

        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.args.t_initial,
            lr_min=0,
            warmup_t=0,
        )
        return [optimizer], [lr_scheduler]


def train_seq_predictor(args: Args):
    L.seed_everything(args.seed)

    if is_wandb_running():
        wandb.init(project="metal-binding")
        args.__dict__.update(dict(wandb.config))

    model = SeqPredictor(args)
    dm = SeqDataModule(args)

    devices = 1
    # logger = L.pytorch.loggers.WandbLogger(project="metal-binding")
    # devices = torch.cuda.device_count()
    logger = None

    # strategy = 'ddp'if devices > 1 else "auto"
    strategy = DDPStrategy(
        find_unused_parameters=True) if devices > 1 else "auto"

    args.__dict__.update(
        {
            "batch_size": args.batch_size // devices,
            "dev_count": devices,
            "t_initial": args.epochs * len(dm.train_dataloader()),
        }
    )

    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f"biolip_{args.metal}_{args.plm}_{args.hidden_size}_{args.num_layers}_" +
        "{epoch}_" + "{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_weights_only=True,
    )
    early_stop = L.pytorch.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=args.patience,
        mode="max",
    )

    precision = "16-mixed" if args.amp else "32-true"

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=5,
        callbacks=[checkpoint, early_stop],
        logger=logger,
    )

    if hasattr(args, 'ckpt') and args.ckpt:
        model.load_state_dict(
            torch.load(args.ckpt)['state_dict'], strict=False)

    trainer.fit(model, dm)

    trainer.validate(model, dm.val_dataloader(),
                     ckpt_path="best", verbose=True)
    time.sleep(1)
    val_test_metrics = trainer.callback_metrics.copy()
    trainer.validate(model, dm.test_dataloader(),
                     ckpt_path="best", verbose=True)
    time.sleep(1)
    val_test_metrics.update(
        [(k.replace("val_", "test_"), v)
         for k, v in trainer.callback_metrics.items()])
    if is_wandb_running():
        wandb.log(val_test_metrics)


class ProbeModel(nn.Module):
    def __init__(self, cnn_layers: int):
        super(ProbeModel, self).__init__()
        self.resnet = generate_model(
            model_depth=cnn_layers,
            n_classes=5,
            num_bins=0.3,
            grid_dim=60,)
        self.fc = nn.Linear(2, 2)

    def forward(self, x, kskp):
        x = self.resnet(x)
        x[..., :2] = x[..., :2] + self.fc(kskp)
        return x


class ProbePredictor(L.LightningModule):
    def __init__(self, args: Args):
        super(ProbePredictor, self).__init__()
        self.args = args

        self.model = generate_model(
            args.cnn_layers,
            n_classes=5,
            num_bins=0.3,
            grid_dim=60,
        )

        self.automatic_optimization = False

        self.cifs = {}
        self.cif_mates = pickle.load(open('data/biolip/cif_mates.pkl', 'rb'))

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        pass

    @line_profiler.profile
    def preload_cifs(self, seqids):
        preloads = []
        for idx, seqid in enumerate(seqids):
            if seqid[:4] not in self.cifs:
                if self.cif_mates.get(seqid[:4], 'Not existed') == 'Success':
                    p = f'data/fixed_cifs/{seqid[:4]}/{seqid[:4]}.cif'
                else:
                    p = f'data/cifs/{seqid[:4]}.cif'
                atoms = get_structure(CIFFile.read(p))[0]

                metal_atoms = atoms[atoms.res_name == self.args.metal]
                atoms = atoms[
                    np.isin(atoms.element, list(elem2token.keys())) &
                    (atoms.hetero == False)]

                coords = torch.tensor(atoms.coord, device='cpu')
                elems = torch.tensor(
                    [elem2token[e] for e in atoms.element], dtype=torch.long, device='cpu')
                metal_coords = torch.tensor(
                    metal_atoms.coord, device='cpu')

                self.cifs[seqid[:4]] = coords, elems, metal_coords
            else:
                coords, elems, metal_coords = self.cifs[seqid[:4]]

            preloads.append((coords, elems, metal_coords, torch.ones(
                len(elems), dtype=torch.long, device='cpu') * idx))

        return preloads

    @line_profiler.profile
    def step(self, batch):
        dev = self.device
        seqids, probes, offsets, kskps = zip(*batch)
        coords, elems, metal_coords, indices = \
            zip(*self.preload_cifs(seqids))

        # probes: [batch_size, 3], offsets: [batch_size, 3]
        # coords: list of [n_atoms, 3], elems: list of [n_atoms]
        coords = torch.cat([
            c.to(dev) - p.to(dev) for c, p in zip(coords, probes)
        ])

        elems = torch.cat(elems).to(dev)
        offsets = torch.stack(offsets).to(dev)
        indices = torch.cat(indices).to(dev)
        probes = torch.stack(probes).to(dev)

        if self.args.use_aug:
            rot = rand_rot(dev).to(probes[0].dtype)
            coords = coords @ rot
            offsets = offsets @ rot

        dist = torch.norm(coords, p=float('inf'), dim=-1)
        mask = dist < self.args.cutoff
        elems = elems[mask]
        coords = coords[mask]
        indices = indices[mask]

        d = torch.norm(offsets, dim=-1)

        pos_labels = d <= self.args.max_offset

        labels = torch.zeros(len(d), device=dev, dtype=torch.long)
        labels[pos_labels] = 1

        x = batch_data(indices, elems, coords)

        # y_pred = self.model(x, torch.stack(kskps).to(dev))
        y_pred = self.model(x)

        indices = indices.unique()
        offsets = offsets[indices]
        labels = labels[indices]
        y_pred = y_pred[indices]
        seqids = [seqids[i] for i in indices]
        probes = probes[indices]

        y_pred_labels = y_pred[..., :2]
        y_pred_offsets = y_pred[..., 2:5]

        if self.args.loss == 'ce':
            loss_labels = F.cross_entropy(y_pred_labels, labels.long())
        elif self.args.loss == 'hard_mining':
            pos_mask = labels == 1
            neg_mask = ~pos_mask
            loss = F.cross_entropy(
                y_pred_labels, labels.long(), reduction='none')
            pos_loss, neg_loss = loss[pos_mask], loss[neg_mask]
            # choose the top 20% of difficult negative samples
            num_hard_neg = int(len(neg_loss) * 0.2)
            hard_neg_loss, _ = torch.topk(neg_loss, num_hard_neg)
            loss_labels = torch.cat([pos_loss, hard_neg_loss]).mean()
        elif self.args.loss == 'fp_penalty':
            # True: normal weights
            # False: more penalized
            pos_mask = labels == 1
            neg_mask = ~pos_mask
            loss = F.cross_entropy(
                y_pred_labels, labels.long(), reduction='none')
            pos_loss, neg_loss = loss[pos_mask], loss[neg_mask]
            loss_labels = torch.cat([
                pos_loss,
                neg_loss * self.args.neg_penalty
            ]).mean()
        else:
            raise NotImplementedError(
                f"Loss {self.args.loss} not implemented")

        if labels.sum() == 0:
            loss_offsets = 0.0
        else:
            loss_offsets = F.smooth_l1_loss(
                y_pred_offsets[labels], offsets[labels])

        return y_pred_offsets, y_pred_labels, labels, offsets, loss_labels, loss_offsets

    def training_step(self, batch, batch_idx):
        _, _, _, _, loss_labels, loss_offsets = self.step(
            batch)

        loss = loss_labels + loss_offsets

        self.log_dict({
            'tr/loss_lbl': loss_labels,
            'tr/loss_off': loss_offsets,
            "tr/loss": loss,
            "cifs": len(self.cifs),
        },
            prog_bar=True)

        self.manual_backward(loss)
        self.optimizers().step()
        self.lr_schedulers().step(self.global_step)
        self.optimizers().zero_grad()

    def validation_step(self, batch, batch_idx):
        ypo, ypl, l, o, _, _ = self.step(batch)
        self.ypos.append(ypo)
        self.ypls.append(ypl)
        self.ls.append(l)
        self.os.append(o)
        pass

    def on_validation_epoch_start(self):
        self.ypos, self.ypls, self.ls, self.os = [], [], [], []

    def on_validation_epoch_end(self):
        ypos = torch.cat(self.ypos)
        ypls = torch.cat(self.ypls)
        ls = torch.cat(self.ls)
        os = torch.cat(self.os)

        # Classification metrics
        try:
            auc = roc_auc_score(ls.cpu(), ypls.softmax(-1).cpu()[..., 1])
        except ValueError:
            auc = 0.0

        f1 = f1_score(ls.cpu(), ypls.argmax(-1).cpu())
        mcc = matthews_corrcoef(ls.cpu(), ypls.argmax(-1).cpu())
        pre = precision_score(ls.cpu(), ypls.argmax(-1).cpu())
        rec = recall_score(ls.cpu(), ypls.argmax(-1).cpu())

        # Regression metric
        rmse = F.mse_loss(ypos[ls], os[ls], reduction='mean').sqrt()
        mae = F.l1_loss(ypos[ls], os[ls], reduction='mean')
        self.log_dict(
            {
                "val_auc": auc,
                "val_f1": f1,
                "val_rmse": rmse,
                "val_mae": mae,
                "val_mcc": mcc,
                "val_pre": pre,
                "val_rec": rec,
            },
            sync_dist=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )
        # 10% of total steps for warmup
        warmup_steps = round(self.args.t_initial * 0.1)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.args.t_initial,
            lr_min=1e-5,
            warmup_t=warmup_steps,
            warmup_lr_init=1e-5,
            warmup_prefix=True,
        )
        return [optimizer], [lr_scheduler]


def train_probe_predictor(args: Args):
    args.update(f'configs/seq_predictors/{args.metal}.yaml')

    L.seed_everything(args.seed)
    if is_wandb_running():
        wandb.init(project="metal-binding")
        args.__dict__.update(dict(wandb.config))

    dm = ProbeDataModule(args)
    model = ProbePredictor(args)

    print(
        f'Data loaded: {len(dm.train_dataloader())} train, {len(dm.val_dataloader())} val, {len(dm.test_dataloader())} test')

    if args.use_pretrained:
        ckpts = {
            10: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=10_val_acc=0.6510.ckpt',
            18: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=18_val_acc=0.6130.ckpt',
            34: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=34_val_acc=0.6550.ckpt',
            50: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=50_val_acc=0.7020.ckpt',
            101: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=101_val_acc=0.7040.ckpt',
            152: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=152_val_acc=0.7100.ckpt',
            200: 'checkpoints/pretrained_models/pretrain_epoch=999_cnn_layers=200_val_acc=0.7620.ckpt',
        }
        args.pretrained_ckpt = ckpts[args.cnn_layers]
        states = torch.load(args.pretrained_ckpt)['state_dict']
        # remove keys starting with "model.fc."
        states = {
            k: v for k, v in states.items() if not k.startswith("model.fc.")}
        model.load_state_dict(states, strict=False)
    else:
        # set learning rate 1e-3
        args.lr = 1e-3  # train from scratch

    if args.metal not in abundants:  # also increase epochs for rare metals
        args.patience = 20
        args.epochs = 200

    devices = 1
    logger = None

    # devices = torch.cuda.device_count()
    # logger = L.pytorch.loggers.WandbLogger(project="metal-binding")

    # strategy = DDPStrategy(
    # find_unused_parameters=True) if devices > 1 else "auto"
    strategy = 'ddp' if devices > 1 else "auto"
    args.__dict__.update(
        {
            "batch_size": args.batch_size // devices,
            "dev_count": devices,
            "t_initial": args.epochs * len(dm.train_dataloader()),
        })
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f'probe_{args.metal}_resnet{args.cnn_layers}_pretrained={args.use_pretrained}_{args.loss}'+"_{epoch}",
        save_weights_only=True,
        # following parameters are used for saving checkpoints every epoch
        save_top_k=-1,
        every_n_epochs=1,
    )
    # early_stop = L.pytorch.callbacks.EarlyStopping(
    #     monitor=monitor,
    #     patience=args.patience,
    #     mode=mode,
    # )
    precision = "16-mixed" if args.amp else "32-true"
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=10,
        callbacks=[
            checkpoint, lr_monitor,  # early_stop
        ],
        logger=logger,
    )

    rich.inspect(args)
    trainer.fit(model, dm)

    val_metrics = trainer.callback_metrics.copy()

    if is_wandb_running():
        wandb.log(val_metrics)


class PretrainModel(L.LightningModule):
    def __init__(self, args: Args):
        super(PretrainModel, self).__init__()
        self.args = args
        self.model = generate_model(
            model_depth=args.cnn_layers,
            n_classes=20,
            num_bins=0.3,
            grid_dim=60,
        )
        self.automatic_optimization = False
        self.cifs = {}

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        data = []
        for i, (pdbid, atoms) in enumerate(batch):
            if pdbid[:4] not in self.cifs:
                ca_idx = np.where((atoms.atom_name == "CA") & np.isin(
                    atoms.res_name, list(_3to1.keys())))[0]
                start_idx = np.where(np.diff(atoms.res_id))[0] + 1
                coords = torch.tensor(atoms.coord, device='cpu')
                elems = torch.tensor(
                    [elem2token[e] for e in atoms.element], dtype=torch.long, device='cpu')
                self.cifs[pdbid[:4]] = (
                    ca_idx, start_idx, coords, elems)
            else:
                ca_idx, start_idx, coords, elems = self.cifs[pdbid[:4]]

            cur_idx = np.random.choice(ca_idx)
            center = coords[cur_idx].to(self.device)
            coords = coords.to(self.device) - center.view(-1, 3)
            dist = torch.norm(coords, p=float('inf'), dim=-1)
            mask = dist < 10
            # find the max j in start_idx where j <= cur_idx, assuming start_idx is sorted
            b = np.searchsorted(start_idx, cur_idx, side='right')
            if b == 0:
                mask[:start_idx[b]] = False
            elif b == len(start_idx):
                mask[start_idx[b - 1]:] = False
            else:
                mask[start_idx[b - 1]:start_idx[b]] = False

            if ms := mask.sum():
                data.append((
                    torch.tensor([i] * ms).to(self.device),
                    elems.to(self.device)[mask], coords[mask],
                    torch.tensor([res2token[_3to1[atoms[cur_idx].res_name]]]).to(
                        self.device)))

        batch_indices, elems, coords, y = zip(*data)
        batch = torch.cat(batch_indices).to(dtype=torch.bfloat16)
        elem = torch.cat(elems).to(dtype=torch.bfloat16)
        coord = torch.cat(coords).to(dtype=torch.bfloat16)
        x = batch_data(batch, elem, coord)
        y = torch.tensor(y).to(self.device)

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch)
        self.log_dict({
            "train/loss": loss,
            "cifs": len(self.cifs),
        }, prog_bar=True)

        self.manual_backward(loss)

        # no gradient accumulation
        self.optimizers().step()
        self.lr_schedulers().step(self.global_step)
        self.optimizers().zero_grad()

    def on_validation_epoch_start(self):
        self.probs, self.labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch)
        self.probs.append(y_hat)
        self.labels.append(y)

    def on_validation_epoch_end(self):
        probs = torch.cat(self.probs)
        labels = torch.cat(self.labels)
        acc = (probs.argmax(-1) == labels).float().mean()
        self.log_dict(
            {
                "val_acc": acc,
            },
            sync_dist=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )

        # 10% of total steps for warmup
        warmup_steps = round(self.args.t_initial * 0.1)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.args.t_initial,
            lr_min=1e-5,
            warmup_t=warmup_steps,
            warmup_lr_init=1e-5,
            warmup_prefix=True,
        )
        return [optimizer], [lr_scheduler]


def pretrain(args: Args):
    L.seed_everything(args.seed)
    if is_wandb_running():
        wandb.init(project="metal-binding")
        args.__dict__.update(dict(wandb.config))

    model = PretrainModel(args)
    dm = wwPDB(args)
    print(
        f'Data loaded: {len(dm.train_dataloader())} train, {len(dm.val_dataloader())} val, {len(dm.test_dataloader())} test')

    devices = 1
    logger = None

    # devices = torch.cuda.device_count()
    logger = L.pytorch.loggers.WandbLogger(project="metal-binding")

    strategy = "ddp" if devices > 1 else "auto"

    steps_per_epoch = len(dm.train_dataloader())
    args.__dict__.update(
        {
            "batch_size": args.batch_size // devices,
            "dev_count": devices,
            "t_initial": args.epochs * steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
        }
    )
    print(f"Total steps: {args.t_initial}")

    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename=f"pretrain_" +
        "{epoch}_"+f"cnn_layers={args.cnn_layers}_" + "{val_acc:.4f}",
        # monitor="val_acc",
        # mode="max",
        # save_weights_only=True,
    )
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step")

    precision = "16-mixed" if args.amp else "32-true"

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        log_every_n_steps=100,
        callbacks=[
            checkpoint, lr_monitor
        ],
        logger=logger,
    )

    # load checkpoint when accidentally interrupted
    # ckpts = list(Path('checkpoints').glob(
    #     f'pretrain_*cnn_layers={args.cnn_layers}*.ckpt'))
    # if len(ckpts) == 1:
    #     args.ckpt_path = str(ckpts[0])

    ckpt_path = args.ckpt_path if hasattr(args, 'ckpt_path') else None
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    trainer.validate(model, dm.val_dataloader(),
                     ckpt_path="best", verbose=True)
    time.sleep(1)
    val_test_metrics = trainer.callback_metrics.copy()
    trainer.validate(model, dm.test_dataloader(),
                     ckpt_path="best", verbose=True)

    time.sleep(1)
    val_test_metrics.update(
        [(k.replace("val_", "test_"), v)
         for k, v in trainer.callback_metrics.items()]
    )

    if is_wandb_running():
        wandb.log(val_test_metrics)
