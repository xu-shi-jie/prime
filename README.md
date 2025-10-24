<div align="center">

# Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations

[Shijie Xu](https://orcid.org/0000-0001-6974-353X), [Akira Onoda](https://orcid.org/0000-0002-5791-4386)

[![BioRxiv](https://img.shields.io/badge/BioRxiv-2025.10.04.680417-red)](https://www.biorxiv.org/content/10.1101/2025.10.04.680417)
[![GitHub](https://img.shields.io/github/stars/xu-shi-jie/prime?style=social)](https://github.com/xu-shi-jie/prime)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-blue)](LICENSE)
</div>

## 🛠️ Installation
For installation, we recommend using [Docker](https://docs.docker.com/engine/install/).
```bash
git clone https://github.com/xu-shi-jie/prime.git
cd prime
docker build -t prime .
docker run --rm -it -d -m 50g --runtime=nvidia --gpus all --shm-size=8g -v $(pwd):/workspace --name prime-dev prime bash
```
You can type `docker exec -it prime-dev bash` to enter the container.

Here is the directory structure of this repository. The model checkpoints are very large, please be patient when downloading.
```bash
├── .dockerignore [0.00 MB]
├── .git [7.15 GB]
├── .gitattributes [0.00 MB]
├── .gitignore [0.00 MB]
├── Dockerfile [0.00 MB]
├── LICENSE [0.02 MB]
├── README.md [0.00 MB]
├── benchmark [4.44 GB]
├── checkpoints [2.08 GB] (* Sequence / pre-trained model checkpoints)
├── configs [0.00 MB]
├── data [524.93 MB]
├── examples [332.23 MB]
├── generate_mates.py [0.00 MB]
├── metal3d_remove_occ.py [0.00 MB]
├── models [0.25 MB]
├── notebooks [49.43 MB]
├── predict.py [0.02 MB]
├── predict_all.py [0.00 MB]
├── predict_probe.py [0.02 MB]
├── requirements.txt [0.00 MB]
├── scripts [0.06 MB]
├── train.py [0.00 MB]
└── weights [6.14 GB] (* Structure model checkpoints)
```

## 🚀 Usage
In Docker container, use predict.py to predict the metal-binding sites of proteins. For example:
```bash
curl -o 12ca.cif https://files.rcsb.org/download/12CA.cif
python predict.py --pdb 12ca.cif:A --outdir . --ckpt weights/probe_ZN_resnet152_pretrained\=True_hard_mining_epoch\=26_val_auc\=0.9910.ckpt
```
where `A` indicates the chain ID of the protein. This will generate PDB files in the output directory, which contain only the predicted metal ions, with naming format as `<file_name>-prime-<metal_type>.pdb`. You can also pass multiple PDB/CIF files at once, as `--pdb file1.cif:A file2.cif:B ...`.

For more details, you can check the help message.
```bash
python predict.py --help
```
We recommend using GPU with enough RAM for the prediction. PRIME was currently tested with NVIDIA GeForce RTX 3090Ti/4090. The default batch size 32 is set for inference of probes. Program running on GPUs with less RAM may reduce the batch size to avoid OOM error.

We also provde script for the prediction of all 14 metal ions:
```bash
python predict_all.py -i 12ca.cif:A -o <outdir>
```

> [!NOTE]
> By default, the prediction program use `torch.compile` to accelerate the inference, which could be slow when the inputs size are small. You can disable it by setting `--no_compile` flag.

> [!WARNING]
> This program does not symmetrize the input protein structures. If you want to do so, please consider using other tools such as PyMOL or ChimeraX to generate the symmetric structures. Names of all generated chains should start with 'sym' prefix.

> [!WARNING]
> DO NOT change the names of checkpoint files, as they contain metadata information used in the prediction program.

## 🧬 ResNet3D representations
We provide pre-trained ResNet3D models for extracting structure representations of proteins. You can use the following code to extract the features.
```python
import torch
from models.resnet import generate_model

# [10, 18, 34, 50, 101, 152, 200]
model = generate_model(10)
model.load_state_dict(torch.load('checkpoints/pretrained/pretrain_epoch=999_cnn_layers=10_val_acc=0.6510.ckpt'))
```
You may need to change the architecture of resnet in `models/resnet.py` to adapt to different tasks.

## 🎓 Citation
```bibtex
@article{xu2025probe,
  title={Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations},
  author={Xu, Shijie and Onoda, Akira},
  journal={bioRxiv},
  pages={2025--10},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## 🌐 Freely available server
Our PRIME server will be soon available at [PRIME Server](https://onodalab.ees.hokudai.ac.jp/prime/).


## ✉️ Help
If you have any questions, please feel free to contact me at `shijie.xu@ees.hokudai.ac.jp`.