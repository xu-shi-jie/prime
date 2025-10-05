# Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations

This repository contains the official implementation of the paper "Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations".

## Installation

It is recommended to use [Docker](https://docs.docker.com/engine/install/) for installation. Ensure that the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed. Build and run the Docker image with the following commands:
```bash
git clone https://github.com/xu-shi-jie/prime.git
cd prime
docker build -t prime .
docker run --rm -it -d -m 50g --runtime=nvidia --gpus all --shm-size=8g -v $(pwd):/workspace --name prime-dev prime bash
```
Access the container using:
```bash
docker exec -it prime-dev bash
```

## Usage

Within the Docker container, use `predict.py` to identify metal-binding sites in proteins. For example:
```bash
curl -o 12ca.cif https://files.rcsb.org/download/12CA.cif
python predict.py --pdb 12ca.cif:A --outdir . --ckpt <ckpt_path>
```
Here, `A` denotes the chain identifier. The output directory will contain PDB files with predicted metal ions, named as `<file_name>-prime-<metal_type>.pdb`. Multiple PDB or CIF files can be processed simultaneously:
```bash
--pdb file1.cif:A file2.cif:B ...
```

Model checkpoints are available in the `weights` directory, with metal types indicated in the filenames. To predict all 14 metal ions concurrently, use:
```bash
python predict_all.py -i 12ca.cif -o .
```

For further information, consult the help message:
```bash
python predict.py --help
```
A GPU with sufficient memory is recommended for prediction. PRIME has been tested with NVIDIA GeForce RTX 3090Ti and 4090. The default batch size for probe inference is 32. On GPUs with limited memory, reduce the batch size to prevent out-of-memory errors.

A script for predicting all 14 metal ions is also provided:
```bash
python predict_all.py -i 12ca.cif:A -o <outdir>
```

> [!NOTE]  
> By default, the prediction program utilizes `torch.compile` to accelerate inference, which may be slower for small input sizes. Disable this feature by setting the `--no_compile` flag.

> [!WARNING]
> This program does not symmetrize input protein structures. To generate symmetric structures, use external tools such as PyMOL or ChimeraX. All generated chain names should begin with the 'sym' prefix.

## Freely Available Server

The PRIME server will be available soon at [PRIME Server](https://onodalab.ees.hokudai.ac.jp/prime/).

## Help

For inquiries, contact `shijie.xu@ees.hokudai.ac.jp`.
