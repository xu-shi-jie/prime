""" refer to: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py"""


from einops import repeat
from functools import partial
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_sum_coo, segment_coo


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=4,
        num_atoms=4,
        num_bins: float = 0.3,
        grid_dim: int = 60,
        sigma=0.6,
    ):
        super().__init__()

        self.num_atoms = num_atoms
        # n_input_channels = num_atoms + 2  # atom type, solvent accessibility, partial charge
        n_input_channels = num_atoms  # atom type
        self.num_bins = num_bins
        self.grid_dim = grid_dim
        self.sigma = sigma
        self.sigma_p = sigma * num_bins

        self.grids = torch.stack(torch.meshgrid(
            self.linspace(),
            self.linspace(),
            self.linspace(),
            indexing="ij",
        ), dim=-1).reshape(-1, 3)

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes -
            out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.to(x.device)

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, tuning='full'):
        x = self.gaussian_blurring(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if tuning == 'partial':
            x = x.detach()

        x = self.fc(x)
        return x

    def gaussian_blurring(self, x: torch.Tensor):
        """Gaussian blurring of the input density map

        Parameters
        ----------
        x: torch.Tensor (n_atoms, 5)

        col 0: batch index
        col 1: atom type
        col 2, 3, 4: x, y, z coordinates

        Returns
        -------
        torch.Tensor (-1, grid_dim, grid_dim, grid_dim)
        """
        # current_batch_size = torch.unique(x[:, 0].long()).shape[0]
        current_batch_size = x[:, 0].max().long() + 1
        fields = torch.zeros(
            (
                current_batch_size,
                self.num_atoms,
                self.grid_dim ** 3,
            ),
            dtype=torch.bfloat16,
            device=x.device,)
        factor = 1.0/(2 * self.sigma_p ** 2)
        for j in range(self.num_atoms):
            atom_j = x[x[:, 1] == j]
            if atom_j.shape[0] > 0:
                density = exp_cdist(self.grids.to(
                    x.device), atom_j[:, 2:5], factor)
                ranges_i = get_ranges(atom_j[:, 0].long())
                fill_density(fields, density, ranges_i, j)

        return fields.view(-1, self.num_atoms, self.grid_dim, self.grid_dim, self.grid_dim)

    def linspace(self):
        return torch.linspace(
            start=-self.grid_dim / 2 * self.num_bins + self.num_bins / 2,
            end=self.grid_dim / 2 * self.num_bins - self.num_bins / 2,
            steps=self.grid_dim,
            dtype=torch.bfloat16,
        )


@torch.compile
def exp_cdist(x1: torch.Tensor, x2: torch.Tensor, p: float):
    density = torch.exp(-torch.cdist(x1, x2) ** 2 * p)
    F.normalize(density, p=1, dim=0, eps=1e-6, out=density)
    return density


@torch.compile
def get_ranges(batch_idx: torch.Tensor):
    return torch.cat([
        torch.tensor([0], device=batch_idx.device, dtype=torch.long),
        torch.where(torch.diff(batch_idx))[0] + 1,
        torch.tensor([
            batch_idx.shape[0]],
            device=batch_idx.device, dtype=torch.long)
    ])


@torch.compile
def fill_density(fields: torch.Tensor, density: torch.Tensor, ranges_i: torch.Tensor, j: int):
    for i in range(ranges_i.shape[0] - 1):
        fields[i, j] = torch.sum(
            density[:, ranges_i[i]:ranges_i[i+1]], dim=1)


def generate_model(model_depth, **kwargs):
    """Generate the model with layers: 10, 18, 34, 50, 101, 152, 200"""
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets
    https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b'''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' \
            else torch.sum(loss) if self.reduction == 'sum' \
            else loss


@torch.compile
def batch_data(indices, elems, coords):
    return torch.cat([
        indices.unsqueeze(-1),
        elems.unsqueeze(-1),
        coords], dim=-1)
