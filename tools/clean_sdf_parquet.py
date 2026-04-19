from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def clean_sdf_parquet(
    input_path: str,
    output_path: str,
    z_percentile: float = 95.0,
    sdf_abs_max: float = 0.05,
    density_percentile: float = 80.0,
    n_neighbors: int = 10,
) -> None:
    df = pd.read_parquet(input_path)

    required_cols = {"x", "y", "z", "s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    sdf = df["s"].to_numpy(dtype=np.float32)

    print(f"\nLecture: {input_path}")
    print("Nombre de points avant filtrage :", len(df))

    # 1) filtre hauteur
    z_limit = np.percentile(points[:, 2], z_percentile)
    mask_height = points[:, 2] < z_limit

    # 2) filtre SDF extrêmes
    mask_sdf = np.abs(sdf) < sdf_abs_max

    # 3) filtre densité locale
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_dist = distances.mean(axis=1)
    density_threshold = np.percentile(mean_dist, density_percentile)
    mask_density = mean_dist < density_threshold

    # combine
    mask = mask_height & mask_sdf & mask_density

    df_clean = df.loc[mask].copy()

    print("z_limit :", z_limit)
    print("density_threshold :", density_threshold)
    print("Nombre de points après filtrage :", len(df_clean))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output, index=False)

    print(f"Fichier sauvegardé : {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin du parquet source")
    parser.add_argument("--output", required=True, help="Chemin du parquet nettoyé")
    parser.add_argument("--z-percentile", type=float, default=95.0)
    parser.add_argument("--sdf-abs-max", type=float, default=0.05)
    parser.add_argument("--density-percentile", type=float, default=80.0)
    parser.add_argument("--n-neighbors", type=int, default=10)
    args = parser.parse_args()

    clean_sdf_parquet(
        input_path=args.input,
        output_path=args.output,
        z_percentile=args.z_percentile,
        sdf_abs_max=args.sdf_abs_max,
        density_percentile=args.density_percentile,
        n_neighbors=args.n_neighbors,
    )


if __name__ == "__main__":
    main()