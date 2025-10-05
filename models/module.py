import torch.nn as nn
from typing import List
import torch
from einops import rearrange
import e3nn


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        # norm: nn.Module = nn.BatchNorm1d,
        norm: nn.Module = None,
        act: nn.Module = nn.LeakyReLU,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if norm is not None:
                self.layers.append(norm(dims[i + 1]))
            self.layers.append(act())
        self.layers.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Header(nn.Module):
    def __init__(
        self, in_size: int, hidden_size: int, num_layers: int, num_classes: int,
        net_type: str = "lstm"
    ):
        super(Header, self).__init__()
        self.net_type = net_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if net_type == "lstm":
            self.lstm = nn.LSTM(
                in_size, hidden_size, num_layers, batch_first=True, bidirectional=True
            )
        elif net_type == "gru":
            self.lstm = nn.GRU(
                in_size, hidden_size, num_layers, batch_first=True, bidirectional=True
            )
        elif net_type == "transformer":
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                batch_first=True
            )
            self.lstm = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.TransformerEncoder(
                    encoder_layers, num_layers=num_layers)
            )
        else:
            raise ValueError(f"Invalid net_type: {net_type}")

        if net_type != "transformer":
            hidden_size = hidden_size * 2
        self.shared_fc = nn.Linear(hidden_size, num_classes)
        self.ion_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, train_shared: bool = False):
        if train_shared:
            if self.net_type == "transformer":
                out = self.lstm(x)
            else:
                out, _ = self.lstm(x)
            return self.shared_fc(out)
        else:
            # with torch.no_grad():
            if self.net_type == "transformer":
                out = self.lstm(x)
            else:
                out, _ = self.lstm(x)
            shared = self.shared_fc(out)
            ion = self.ion_fc(out)
            return shared + ion


class ConvFCLayers(nn.Module):
    def __init__(self, in_size: int, num_classes: int):
        super(ConvFCLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_size, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_repr: bool = False):
        x = self.layers(rearrange(x, "b l c -> b c l"))
        if return_repr:
            return x
        return self.fc(rearrange(x, "b c l -> b l c"))


class Conv3dNet(nn.Module):
    """3D convolutional neural network,
    refer to: https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg/blob/9ddca83765950edfd05ee2e0ad4003590b42dfe3/src/rasp_model.py#L223
    """

    def __init__(
        self,
        n_layers: int = 3,
        num_atoms: int = 6,
        num_bins: float = 1.0,
        grid_dim: int = 18,
        sigma=0.6,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_bins = num_bins
        self.grid_dim = grid_dim
        self.sigma = sigma
        self.sigma_p = sigma * num_bins

        self.xx, self.yy, self.zz = torch.meshgrid(
            self.linspace(),
            self.linspace(),
            self.linspace(),
            indexing="ij",
        )
        layers = []
        in_dim, out_dim = 6, 16
        for i in range(n_layers):
            layers.extend(
                [
                    nn.Conv3d(
                        in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=1
                    ),
                    nn.MaxPool3d(
                        kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)
                    ),
                    nn.BatchNorm3d(out_dim),
                    nn.LeakyReLU(),
                ]
            )
            in_dim = out_dim
            out_dim *= 2
        self.layers = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * out_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 100),
        )
        self.layers = nn.Sequential(*layers)
        self.header = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(100, 20),
        )

    def linspace(self):
        return torch.linspace(
            start=-self.grid_dim / 2 * self.num_bins + self.num_bins / 2,
            end=self.grid_dim / 2 * self.num_bins - self.num_bins / 2,
            steps=self.grid_dim,
            dtype=torch.float32,
        )

    def forward(self, x, return_repr: bool = False):
        x = self.gaussian_blurring(x)
        x = self.layers(x)
        x = self.mlp(x)
        if return_repr:
            return x
        else:
            return self.header(x)

    def gaussian_blurring(self, x: torch.Tensor):
        """Gaussian blurring of the input density map

        Parameters
        ----------
        x: torch.Tensor (n_atoms, 5)

        col 0: batch index
        col 1: atom type
        col 2,3,4: x,y,z coordinates

        Returns
        -------
        torch.Tensor (-1, grid_dim, grid_dim, grid_dim)
        """
        current_batch_size = torch.unique(x[:, 0]).shape[0]
        fields = torch.zeros(
            (
                current_batch_size,
                self.num_atoms,
                self.grid_dim,
                self.grid_dim,
                self.grid_dim,
            )
        ).to(x.device)
        for j in range(self.num_atoms):
            mask_j = x[:, 1] == j
            atom_type_j_data = x[mask_j]
            if atom_type_j_data.shape[0] > 0:
                pos = atom_type_j_data[:, 2:]
                density = torch.exp(
                    -(
                        (torch.reshape(self.xx.to(x.device),
                         [-1, 1]) - pos[:, 0]) ** 2
                        + (torch.reshape(self.yy.to(x.device),
                           [-1, 1]) - pos[:, 1])
                        ** 2
                        + (torch.reshape(self.zz.to(x.device),
                           [-1, 1]) - pos[:, 2])
                        ** 2
                    )
                    / (2 * self.sigma_p**2)
                )

                # Normalize each atom to 1
                density /= torch.sum(density, dim=0) + \
                    1e-6  # avoid division by zero
                change_mask_j = (
                    atom_type_j_data[:, 0][:-1] != atom_type_j_data[:, 0][1:]
                )
                ranges_i = torch.cat(
                    [
                        torch.tensor([0]).to(x.device),
                        torch.arange(atom_type_j_data.shape[0] - 1, device=x.device)[
                            change_mask_j
                        ]
                        + 1,
                        torch.tensor([atom_type_j_data.shape[0]]).to(x.device),
                    ]
                )
                for i in range(ranges_i.shape[0]):
                    if i < ranges_i.shape[0] - 1:
                        index_0, index_1 = ranges_i[i], ranges_i[i + 1]
                        fields[i, j, :, :, :] = torch.reshape(
                            torch.sum(density[:, index_0:index_1], dim=1),
                            [self.grid_dim, self.grid_dim, self.grid_dim],
                        )
        return fields


class DensityNet(nn.Module):
    def __init__(self, n_aa: int, n_atom: int, n_metal: int, n_element: int):
        super().__init__()
        self.res_emb = nn.Embedding(n_aa, 32)
        self.atom_emb = nn.Embedding(n_atom, 64)
        self.element_emb = nn.Embedding(n_element, 8)
        self.metal_emb = nn.Embedding(n_metal, 32)

        self.net = nn.Sequential(
            nn.Linear(1+n_aa+n_atom+n_metal+n_element, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU())

    def forward(self, x):
        return self.net(x)
