from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = {"x", "y", "z", "s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path}: {sorted(missing)}. "
            f"Colonnes trouvées: {list(df.columns)}"
        )

    return df


def filter_by_sdf(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["s"].abs() < threshold].copy()


def compute_edges(df: pd.DataFrame, nx: int, ny: int):
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    return x_edges, y_edges


def extract_tile(
    df: pd.DataFrame,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    include_right: bool = False,
    include_top: bool = False,
) -> pd.DataFrame:
    if include_right:
        x_mask = (df["x"] >= x0) & (df["x"] <= x1)
    else:
        x_mask = (df["x"] >= x0) & (df["x"] < x1)

    if include_top:
        y_mask = (df["y"] >= y0) & (df["y"] <= y1)
    else:
        y_mask = (df["y"] >= y0) & (df["y"] < y1)

    return df.loc[x_mask & y_mask].copy()


def save_tiles(
    df: pd.DataFrame,
    x_edges,
    y_edges,
    out_dir: Path,
    prefix: str,
    min_points: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    for i in range(nx):
        for j in range(ny):
            include_right = (i == nx - 1)
            include_top = (j == ny - 1)

            tile = extract_tile(
                df,
                x_edges[i],
                x_edges[i + 1],
                y_edges[j],
                y_edges[j + 1],
                include_right=include_right,
                include_top=include_top,
            )

            if len(tile) >= min_points:
                tile_path = out_dir / f"{prefix}_tile_{i}_{j}.parquet"
                tile.to_parquet(tile_path, index=False)
                saved.append((i, j, len(tile), tile_path))

    return saved


def main():
    parser = argparse.ArgumentParser(description="Découpage train/val en tuiles parquet pour DeepSDF.")
    parser.add_argument("--train", required=True, help="Parquet train complet")
    parser.add_argument("--val", required=True, help="Parquet val complet")
    parser.add_argument("--threshold", type=float, default=0.005, help="Filtre |s| < threshold")
    parser.add_argument("--nx", type=int, default=4, help="Nombre de tuiles en x")
    parser.add_argument("--ny", type=int, default=4, help="Nombre de tuiles en y")
    parser.add_argument("--min-points", type=int, default=3000, help="Nombre minimal de points par tuile")
    parser.add_argument("--out", required=True, help="Dossier de sortie racine")
    parser.add_argument("--prefix", default="lfpo", help="Préfixe des tuiles")
    args = parser.parse_args()

    train_df = load_df(args.train)
    val_df = load_df(args.val)

    print("Train shape avant filtre:", train_df.shape)
    print("Val shape avant filtre:", val_df.shape)

    train_df = filter_by_sdf(train_df, args.threshold)
    val_df = filter_by_sdf(val_df, args.threshold)

    print("Train shape après filtre:", train_df.shape)
    print("Val shape après filtre:", val_df.shape)

    # Les bornes des tuiles sont calculées à partir du train
    x_edges, y_edges = compute_edges(train_df, args.nx, args.ny)

    out_root = Path(args.out)
    train_out = out_root / "train"
    val_out = out_root / "val"

    saved_train = save_tiles(
        train_df,
        x_edges,
        y_edges,
        train_out,
        args.prefix,
        args.min_points,
    )

    saved_val = save_tiles(
        val_df,
        x_edges,
        y_edges,
        val_out,
        args.prefix,
        args.min_points,
    )

    print("\n--- Tuiles train sauvegardées ---")
    if saved_train:
        for i, j, n, path in saved_train:
            print(f"tile ({i},{j}) -> {n} points -> {path}")
    else:
        print("Aucune tuile train sauvegardée.")

    print("\n--- Tuiles val sauvegardées ---")
    if saved_val:
        for i, j, n, path in saved_val:
            print(f"tile ({i},{j}) -> {n} points -> {path}")
    else:
        print("Aucune tuile val sauvegardée.")

    print("\nTerminé.")


if __name__ == "__main__":
    main()