import torch
import torch.nn as nn


class FourierFeatures(nn.Module):

    def __init__(self, input_dim=3, mapping_size=32, scale=10):#mettre un mzpping size a 128 pour les aeroport 

        super().__init__()
        if mapping_size <= 0: #ajout d'une vérification pour éviter les erreurs de dimension
            raise ValueError("mapping_size doit être > 0")
        
        B = torch.randn((input_dim, mapping_size),dtype=torch.float32) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_proj = 2.0 * torch.pi * x @ self.B

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)