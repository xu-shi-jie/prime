<div align="center">

# Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations

[Shijie Xu](https://orcid.org/0000-0001-6974-353X), [Akira Onoda](https://orcid.org/0000-0002-5791-4386)

[![BioRxiv](https://img.shields.io/badge/BioRxiv-2025.10.04.680417-red)](https://www.biorxiv.org/content/10.1101/2025.10.04.680417)
[![GitHub](https://img.shields.io/github/stars/xu-shi-jie/prime?style=social)](https://github.com/xu-shi-jie/prime)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

[![PyPI](https://img.shields.io/pypi/v/prime-metal?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/prime-metal/)
[![PyPI downloads](https://img.shields.io/pypi/dm/prime-metal?logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypistats.org/packages/prime-metal)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/prime-metal?logo=anaconda&logoColor=white&label=Bioconda)](https://bioconda.github.io/recipes/prime-metal/README.html)
[![Bioconda downloads](https://img.shields.io/conda/dn/bioconda/prime-metal?logo=anaconda&logoColor=white&label=Bioconda%20downloads)](https://anaconda.org/bioconda/prime-metal)
</div>

## 🛠️ Installation
### With pip or conda
```bash
pip install prime-metal
# or
conda install -c conda-forge -c bioconda prime-metal
```
This installs the `prime` command and the `prime` Python package. Model
checkpoints (~6.5 GB) are **not** part of the package: they are downloaded from
the [Hugging Face Hub](https://huggingface.co/xushijie/prime) the first time a
metal is predicted, and cached afterwards. Use `prime download ZN CA` to fetch
them in advance, or set `PRIME_HOME` to a directory that already contains
`weights/` and `checkpoints/` (e.g. a clone of this repository).

PRIME requires a CUDA GPU. Make sure PyTorch is installed with CUDA support
(`conda install -c conda-forge 'pytorch=*=*cuda*'`, or follow
[pytorch.org](https://pytorch.org/get-started/locally/) for pip).

### With Docker (recommended for training and for reproducing the paper)
For installation, we recommend using [Docker](https://docs.docker.com/engine/install/). You also need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support in Docker.
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
├── notebooks [49.43 MB]
├── predict.py [0.00 MB] (* wrapper around prime/predict.py)
├── predict_all.py [0.00 MB]
├── predict_probe.py [0.02 MB]
├── prime [0.36 MB] (* the installable package: models, CLI and configs)
├── requirements.txt [0.00 MB]
├── scripts [0.06 MB]
├── train.py [0.00 MB]
└── weights [6.14 GB] (* Structure model checkpoints)
```

## 🚀 Usage
Use the `prime` command to predict the metal-binding sites of proteins. For example:
```bash
curl -o 12ca.cif https://files.rcsb.org/download/12CA.cif
prime predict --pdb 12ca.cif:A --metal ZN --outdir .
```
Inside a clone of this repository (or in the Docker container) you can equivalently run the script and choose the checkpoint yourself:
```bash
python predict.py --pdb 12ca.cif:A --outdir . --ckpt weights/probe_ZN_resnet152_pretrained\=True_hard_mining_epoch\=26_val_auc\=0.9910.ckpt
```
where `A` indicates the chain ID of the protein. This will generate PDB files in the output directory, which contain only the predicted metal ions, with naming format as `<file_name>-prime-<metal_type>.pdb`. You can also pass multiple PDB/CIF files at once, as `--pdb file1.cif:A file2.cif:B ...`.

For more details, you can check the help message.
```bash
prime predict --help
```
We recommend using GPU with enough RAM for the prediction. PRIME was currently tested with NVIDIA GeForce RTX 3090Ti/4090. The default batch size 32 is set for inference of probes. Program running on GPUs with less RAM may reduce the batch size to avoid OOM error.

We also provde script for the prediction of all 14 metal ions:
```bash
prime predict-all -i 12ca.cif:A -o <outdir>
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
from prime.models.resnet import generate_model

# [10, 18, 34, 50, 101, 152, 200]
model = generate_model(10)
model.load_state_dict(torch.load('checkpoints/pretrained/pretrain_epoch=999_cnn_layers=10_val_acc=0.6510.ckpt'))
```
You may need to change the architecture of resnet in `prime/models/resnet.py` to adapt to different tasks.

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
Our PRIME server is now available at [PRIME Server](https://onodalab.ees.hokudai.ac.jp/prime/).

## 🔍 FAQ
<details>
<summary>I cannot access your PRIME server.</summary>
~~Our server has been updated. If you continue to encounter this issue, please contact us.~~
~~Unfortunately, our server is currently updating its certificate, which means you may encounter security warnings when trying to access it. Some browsers may block access to the site due to these warnings (e.g., Safari). We recommend using browsers like Chrome or Firefox -> Advanced -> Proceed to the site (unsafe). We apologize for any inconvenience this may cause and appreciate your understanding.~~
</details>


## ✉️ Help
If you have any questions, please feel free to contact me at `shijie.xu@ees.hokudai.ac.jp`.