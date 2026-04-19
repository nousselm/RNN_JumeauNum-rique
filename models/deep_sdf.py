#1ere version de deep sdf sans fourier features
#  import torch
# import torch.nn as nn


# class DeepSDF(nn.Module):

#     def __init__(self, hidden_dim=256, n_layers=4):

#         super().__init__()

#         layers = []

#         layers.append(nn.Linear(3, hidden_dim))
#         layers.append(nn.ReLU())

#         for _ in range(n_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())

#         layers.append(nn.Linear(hidden_dim, 1))

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):

#         return self.network(x)
    

import torch
import torch.nn as nn
from models.fourier_features import FourierFeatures


class DeepSDF(nn.Module):

    def __init__(self, hidden_dim=256, n_layers=4, use_fourier=True, fourier_dim=32,fourier_scale: float = 10.0):

        super().__init__()

        self.use_fourier = use_fourier

        if use_fourier:

            self.embed = FourierFeatures(
                input_dim=3,
                mapping_size=fourier_dim,
                scale=fourier_scale,
            )
            input_dim = fourier_dim * 2

        else:
            self.embed = None
            input_dim = 3

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers-1):

            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim,1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_fourier:

            x = self.embed(x)

        return self.network(x)