from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_parquet(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    required_cols = {"x", "y", "z", "s"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)} ; colonnes trouvées : {list(df.columns)}"
        )

    return df


def print_stats(df: pd.DataFrame, name: str = "dataset") -> None:
    print(f"\n=== Stats: {name} ===")
    print("Shape:", df.shape)
    print("Colonnes:", df.columns.tolist())
    print("\nMin:")
    print(df[["x", "y", "z", "s"]].min())
    print("\nMax:")
    print(df[["x", "y", "z", "s"]].max())
    print("\nDescribe:")
    print(df[["x", "y", "z", "s"]].describe())


def filter_near_surface(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    mask = df["s"].abs() < threshold
    return df.loc[mask].copy()


def export_points_obj(df: pd.DataFrame, output_path: str) -> None:
    """
    Exporte un nuage de points minimal en OBJ :
    chaque ligne 'v x y z' = un sommet.
    Blender peut l'importer.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            f.write(f"v {row.x} {row.y} {row.z}\n")

    print(f"OBJ exporté : {output}")


def export_points_csv(df: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"CSV exporté : {output}")


def crop_xy_zone(
    df: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> pd.DataFrame:
    mask = (
        (df["x"] >= x_min)
        & (df["x"] <= x_max)
        & (df["y"] >= y_min)
        & (df["y"] <= y_max)
    )
    return df.loc[mask].copy()


def split_into_xy_tiles(
    df: pd.DataFrame,
    nx: int,
    ny: int,
    min_points: int = 1000,
) -> list[tuple[int, int, pd.DataFrame]]:
    """
    Découpe la scène en tuiles régulières sur (x,y).
    Retourne uniquement les tuiles avec au moins min_points points.
    """
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    tiles: list[tuple[int, int, pd.DataFrame]] = []

    for i in range(nx):
        for j in range(ny):
            tile = crop_xy_zone(
                df,
                x_edges[i],
                x_edges[i + 1],
                y_edges[j],
                y_edges[j + 1],
            )
            if len(tile) >= min_points:
                tiles.append((i, j, tile))

    return tiles


def save_tiles_as_parquet(
    tiles: list[tuple[int, int, pd.DataFrame]],
    output_dir: str,
    prefix: str,
) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, j, tile in tiles:
        path = outdir / f"{prefix}_tile_{i}_{j}.parquet"
        tile.to_parquet(path, index=False)

    print(f"{len(tiles)} tuiles sauvegardées dans {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspection / filtrage / découpage d'un dataset SDF parquet.")
    parser.add_argument("--input", required=True, help="Chemin du fichier parquet")
    parser.add_argument("--threshold", type=float, default=0.01, help="Seuil pour |s| < threshold")
    parser.add_argument("--export-surface-obj", type=str, default="", help="Chemin de l'OBJ des points proches de la surface")
    parser.add_argument("--export-surface-csv", type=str, default="", help="Chemin du CSV des points proches de la surface")
    parser.add_argument("--tile-nx", type=int, default=0, help="Nombre de tuiles en x")
    parser.add_argument("--tile-ny", type=int, default=0, help="Nombre de tuiles en y")
    parser.add_argument("--tile-min-points", type=int, default=1000, help="Nombre minimum de points par tuile")
    parser.add_argument("--tile-output-dir", type=str, default="", help="Dossier de sortie des tuiles parquet")
    parser.add_argument("--tile-prefix", type=str, default="lfpo", help="Préfixe des tuiles")
    args = parser.parse_args()

    df = load_parquet(args.input)
    print_stats(df, name=args.input)

    near_surface = filter_near_surface(df, args.threshold)
    print_stats(near_surface, name=f"near_surface |s|<{args.threshold}")

    if args.export_surface_obj:
        export_points_obj(near_surface, args.export_surface_obj)

    if args.export_surface_csv:
        export_points_csv(near_surface, args.export_surface_csv)

    if args.tile_nx > 0 and args.tile_ny > 0 and args.tile_output_dir:
        tiles = split_into_xy_tiles(
            near_surface,
            nx=args.tile_nx,
            ny=args.tile_ny,
            min_points=args.tile_min_points,
        )
        save_tiles_as_parquet(tiles, args.tile_output_dir, args.tile_prefix)


if __name__ == "__main__":
    main()