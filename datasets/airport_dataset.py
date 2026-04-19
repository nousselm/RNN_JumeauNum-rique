from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class AirportSDFParquetDataset(Dataset):
    def __init__(self, parquet_path: str, clamp: float | None = None, verbose: bool = False):
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)

        required_cols = {"x", "y", "z", "s"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {parquet_path}: {sorted(missing)}. "
                f"Colonnes trouvées: {list(self.df.columns)}"
            )

        if verbose:
            print(f"\nDataset: {parquet_path}")
            print(self.df[["x", "y", "z", "s"]].describe())
            print("Bounds min:")
            print(self.df[["x", "y", "z"]].min())
            print("Bounds max:")
            print(self.df[["x", "y", "z"]].max())

        self.points = torch.tensor(
            self.df[["x", "y", "z"]].values,
            dtype=torch.float32
        )

        sdf = torch.tensor(
            self.df["s"].values,
            dtype=torch.float32
        )

        #sdf = torch.clamp(sdf, -0.05, 0.05) #diff ici

        if clamp is not None:
            sdf = torch.clamp(sdf, -clamp, clamp)

        self.sdf = sdf.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int):
        return self.points[idx], self.sdf[idx]
    


