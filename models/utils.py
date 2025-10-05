import random
import os
import signal

import psutil
import torch
import yaml
from functools import wraps
import errno
import signal
import numpy as np
from scipy.spatial import KDTree
from math import ceil
from tqdm import tqdm
import line_profiler
import math
import cupy as cp
import string


def num_parameters(model: torch.nn.Module) -> int:
    """Return the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Config:
    """Read configuration from a YAML file and store as attributes"""

    def __init__(self, yaml_file: str):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(self, k, v)

    def update(self, new_yaml_file: str):
        with open(new_yaml_file, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(self, k, v)

    def save(self, yaml_file: str):
        with open(yaml_file, "w") as f:
            yaml.dump(self.__dict__, f)


def memory_usage_psutil():
    """Return the memory usage in percentage like top"""
    process = psutil.Process(os.getpid())
    mem = process.memory_percent()
    return mem


def is_wandb_running():
    """Check if wandb is running"""
    return "WANDB_SWEEP_ID" in os.environ


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def shorten_path(path: str, max_len: int = 30) -> str:
    """Shorten the path to max_len characters"""
    if len(path) > max_len:
        return path[:max_len // 2] + "..." + path[-max_len // 2:]
    return path


def cluster_points(data: torch.Tensor, d: float) -> torch.Tensor:
    """
    Cluster points based on the Euclidean distance.

    :param data: Input data, shape (n_points, n_features), type torch.Tensor.
    :param d: Distance threshold for clustering.
    :return: Cluster indices, shape (n_points,), type torch.Tensor.
    """
    dist = torch.cdist(data, data)
    indices = torch.full((data.shape[0],), -1, dtype=torch.long)
    cluster_id = 0
    for i in range(data.shape[0]):
        if indices[i] == -1:
            indices[dist[i] < d] = cluster_id
            cluster_id += 1
    return indices


def bron_kerbosch(R, P, X, graph):
    if not P and not X:
        yield R
    while P:
        v = P.pop()
        yield from bron_kerbosch(
            R | {v},
            P & set(graph[v]),
            X & set(graph[v]),
            graph
        )
        X.add(v)


def find_cliques(graph):
    """
    Find all maximal cliques in an undirected graph with the Bron–Kerbosch algorithm.

    :param graph: Input graph as a NetworkX graph
    :return: List of maximal cliques
    """
    return list(bron_kerbosch(set(), set(graph.nodes()), set(), graph))


def segment_cmd(cmd_str: str, max_len: int = 1000):
    cmds = ['']
    prev = 0
    for i, c in enumerate(cmd_str):
        if c == ';':
            if len(cmds[-1]) + len(cmd_str[prev:i]) > max_len:
                cmds.append('')
            cmds[-1] += cmd_str[prev:i + 1]
            prev = i + 1
    return cmds


def get_color(v):
    assert 0 <= v <= 1, f'v should be in [0, 1], got {v}'
    # green to brown
    color1 = np.array([0, 128, 0])
    color2 = np.array([165, 42, 42])
    v = v * (color2 - color1) + color1
    v /= 255
    return f'[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]'


def generate_pymol_script(possible_sites):
    cmd = ''
    for i, pos in enumerate(possible_sites):
        cmd += f"pseudoatom s{i},pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}];color blue,s{i};"
    return cmd


def remove_close_points_kdtree(points, min_distance):
    tree = KDTree(points)
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(
            point, min_distance)
        keep[neighbors] = False
        keep[i] = True  # Keep the current point
    return points[keep]


@line_profiler.profile
def pack_bit(x: torch.Tensor):
    """ Pack the bit tensor to a sequence of bytes.
    Args:
        x (torch.Tensor): The input tensor to be packed.
    Returns:
        torch.Tensor: The packed tensor.
    """
    batch_size, num_bits = x.shape
    num_bytes = (num_bits + 7) // 8
    output = torch.zeros(batch_size, num_bytes,
                         dtype=torch.uint8, device=x.device)
    for i in range(num_bits):
        byte_index = i // 8
        bit_index = i % 8
        output[:, byte_index] |= (x[:, i] << bit_index).to(torch.uint8)
    return output


@line_profiler.profile
def unpack_bit(x: torch.Tensor, num_bits: int):
    """ Unpack the bit tensor from a sequence of bytes.
    Args:
        x (torch.Tensor): The input tensor to be unpacked.
        num_bits (int): The number of bits to unpack.
    Returns:
        torch.Tensor: The unpacked tensor.
    """
    batch_size, num_bytes = x.shape
    output = torch.zeros(batch_size, num_bits,
                         dtype=torch.uint8, device=x.device)
    for i in range(num_bits):
        byte_index = i // 8
        bit_index = i % 8
        output[:, i] = (x[:, byte_index] >> bit_index) & 1
    return output


def safe_dist(vec1: torch.Tensor, vec2: torch.Tensor, max_size: int = 100_000_000, p: int = 2):
    """ compute the minimum distance between two vectors:

    vec1: (N, 3), N could be very very large, i.e., all atoms' coordinates in a large protein

    vec2: (M, 3), M are not very large, usually the coordinates of the binding sites

    max_size: the maximum size of the distance matrix to compute at once

    p: the p-norm to use for distance calculation

    return: (M, ) the minimum distance of each binding site to the protein
    """
    size1 = vec1.shape
    size2 = vec2.shape
    batch_size = ceil(max_size / size1[0])
    dists = []
    for i in range(0, size2[0], batch_size):
        dist = torch.cdist(vec1, vec2[i:i + batch_size], p=p)
        dists.append(dist.min(dim=0).values)
    return torch.cat(dists)


@line_profiler.profile
def safe_filter(nos: torch.Tensor, pos: torch.Tensor, thr: torch.Tensor, all: torch.Tensor, lb: float, max_size: int = 100_000_000):
    """ filter the binding sites based on the distance matrix 
    nos: (N, 3), N are the coordinates of the binding sites
    *pos: (M, 3), M are the coordinates of the protein, could be very very large
    thr: (N, 2), the distance threshold for each binding site
    all: (P, 3), P are the coordinates of all atoms in the protein
    lb: the lower bound of the distance

    return: (N, M) available binding sites
    """
    N, M, P = nos.shape[0], pos.shape[0], all.shape[0]
    batch_size = ceil(max_size / N)
    output = []
    interests = []
    for i in tqdm(range(0, M, batch_size), leave=False, desc=f'Filtering (batch_size: {batch_size})'):
        dist = torch.cdist(pos[i:i + batch_size], nos)
        dist = (dist <= thr[:, 1].unsqueeze(0)) & \
            (dist >= thr[:, 0].unsqueeze(0))
        dist_all = safe_dist(all, pos[i:i + batch_size]) > lb
        dist = dist & dist_all.unsqueeze(-1)

        mask = dist.any(dim=1)
        output.append(pack_bit(dist[mask]).T)
        interests.append(mask)
    return torch.cat(output, dim=1), torch.cat(interests)


def backbone(atoms, chain_id):
    """ return the atoms of the backbone of a chain """
    return atoms[
        (atoms.chain_id == chain_id) &
        (atoms.atom_name == "CA") &
        (atoms.element == "C")]


def get_color(v):
    assert 0 <= v <= 1, f'v should be in [0, 1], got {v}'
    # green to brown
    color1 = np.array([0, 128, 0])
    color2 = np.array([165, 42, 42])
    v = v * (color2 - color1) + color1
    v /= 255
    return f'[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]'


@line_profiler.profile
@torch.compile()
def kde_pytorch(x, x_samples, bandwidth=0.1, k=None, kernel='gaussian'):
    """
    Kernel density estimation using Gaussian kernel.

    Arguments:
        x: Query point (N,)
        x_samples: Sample points (M,)
        bandwidth: Kernel bandwidth (float)
        k: Number of nearest neighbors (int, optional)
        kernel: Kernel type (str, default 'gaussian', or 'epanechnikov')

    Returns:
        Kernel density estimate (N,)

    """
    x = x.unsqueeze(-1)
    x_samples = x_samples.view(1, -1)
    u = (x - x_samples) / bandwidth
    sq_u = u ** 2

    use_topk = False
    if k is not None and k < x_samples.shape[1]:
        use_topk = True
        topk_vals, _ = torch.topk(sq_u, k=k, dim=1, largest=False)
        u = torch.sign(u) * torch.sqrt(topk_vals)

    if kernel == 'gaussian':
        if use_topk:
            kernels = torch.exp(-0.5*topk_vals) / (bandwidth*(2*np.pi)**0.5)
        else:
            kernels = torch.exp(-0.5*sq_u) / (bandwidth*(2*np.pi)**0.5)
    elif kernel == 'epanechnikov':
        if use_topk:
            mask = (topk_vals <= 1).float()
            kernels = 0.75 * (1 - topk_vals) * mask / bandwidth
        else:
            mask = (sq_u <= 1).float()
            kernels = 0.75 * (1 - sq_u) * mask / bandwidth
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return kernels.mean(dim=1)


@torch.compile()
def log_kde(x: torch.Tensor, samples: torch.Tensor, bandwidth=0.1):
    # x: (N,), samples: (M,)
    diff = x[:, None] - samples[None, :]  # (N, M)
    exponent = -0.5 * (diff / bandwidth)**2
    log_kernel = exponent - math.log(math.sqrt(2 * math.pi) * bandwidth)
    # logsumexp over sample axis
    log_density = torch.logsumexp(
        log_kernel, dim=1) - math.log(samples.shape[0])
    return log_density  # log-density


@torch.compile()
def safe_cdist_thr(x: torch.Tensor, threshold: float, batch_size: int = 1000, ):
    """ safe cdist with threshold
    Arguments:
        x: (N, 3), N could be very very large, i.e., all atoms' coordinates in a large protein
        threshold: the threshold to filter the distance
        batch_size: the batch size to compute the distance matrix

    Returns:
        torch.Tensor: (N, N) the distance matrix
    """
    d = torch.zeros(x.shape[0], x.shape[0], device=x.device, dtype=torch.bool)
    for i in tqdm(range(0, x.shape[0], batch_size), leave=False, desc=f'Computing distance matrix'):
        for j in range(i, x.shape[0], batch_size):
            dist = torch.cdist(x[i:i + batch_size],
                               x[j:j + batch_size]) <= threshold
            d[i:i + batch_size, j:j + batch_size] = dist
            d[j:j + batch_size, i:i + batch_size] = dist.T
    return d


def scatter_medoid(positions: torch.Tensor, indices: torch.Tensor, dim_size: int):
    kernel_code = r'''
    extern "C" {

    __global__
    void cluster_medoid(const float* __restrict__ positions,
                        const long long* __restrict__ indices,
                        float* output,
                        long long* output_indices,
                        int N, int D, int C) {
        int cluster_id = blockIdx.x;
        int tid = threadIdx.x;

        extern __shared__ char shared_mem[];
        float* sum_dists = (float*)shared_mem;
        long long* sum_indices = (long long*)(sum_dists + blockDim.x);

        float my_sum = 1e30f;
        long long my_idx = -1;

        for (int i = tid; i < N; i += blockDim.x) {
            if (indices[i] == cluster_id) {
                float sum_dist = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (indices[j] == cluster_id) {
                        float dist = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            float diff = positions[i * D + d] - positions[j * D + d];
                            dist += diff * diff;
                        }
                        sum_dist += sqrtf(dist);
                    }
                }
                if (sum_dist < my_sum || (fabs(sum_dist - my_sum) < 1e-7 && i < my_idx)) {
                    my_sum = sum_dist;
                    my_idx = i;
                }
            }
        }

        sum_dists[tid] = my_sum;
        sum_indices[tid] = my_idx;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                if (sum_dists[tid + offset] < sum_dists[tid]) {
                    sum_dists[tid] = sum_dists[tid + offset];
                    sum_indices[tid] = sum_indices[tid + offset];
                }
            }
            __syncthreads();
        }

        if (tid == 0 && sum_indices[0] >= 0) {
            int best_idx = sum_indices[0];
            for (int d = 0; d < D; ++d) {
                output[cluster_id * D + d] = positions[best_idx * D + d];
            }
            output_indices[cluster_id] = best_idx;
        }
    }

    } // extern "C"
    '''
    module = cp.RawModule(code=kernel_code)
    cluster_medoid = module.get_function('cluster_medoid')
    N, D = positions.shape
    positions_cp = cp.asarray(positions)
    indices_cp = cp.asarray(indices)
    output_cp = cp.zeros((dim_size, D), dtype=cp.float32)
    output_indices_cp = cp.zeros(dim_size, dtype=cp.int64)
    threads_per_block = 256
    blocks_per_grid = dim_size
    shared_mem = threads_per_block * \
        (cp.dtype(cp.float32).itemsize + cp.dtype(cp.int64).itemsize)

    cluster_medoid(
        (blocks_per_grid,),
        (threads_per_block,),
        (positions_cp, indices_cp, output_cp, output_indices_cp, N, D, dim_size),
        shared_mem=shared_mem)
    output = torch.as_tensor(output_cp, device=positions.device)
    output_indices = torch.as_tensor(
        output_indices_cp, device=positions.device)
    return output, output_indices


def generate_id(k: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=k))
