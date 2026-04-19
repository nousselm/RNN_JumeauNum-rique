import torch
from torch.utils.data import Dataset

#V1 inchangé, V2 inchangé, V3 j'ai ajouté un clamp pour limiter les valeurs de sdf et éviter que l'image du cube soit trop bruitée, V4 j'ai ajouté des points proches de la surface et loin de la surface pour améliorer l'apprentissage du modèle
class SDFDataset(Dataset):

    def __init__(self, points, sdf,clamp=0.1): #j'ai ajouter clamp car l'image du cube etaut tres bruité
        sdf = torch.clamp(sdf, -clamp, clamp)  #de meme V3
        
        self.points = points.float()
        self.sdf = sdf.unsqueeze(1).float()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):

        return self.points[idx], self.sdf[idx]