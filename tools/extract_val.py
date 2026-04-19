from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tile", required=True, help="Chemin de la tuile train existante")
    parser.add_argument("--val-full", required=True, help="Chemin du parquet val complet")
    parser.add_argument("--output", required=True, help="Chemin de sortie de la tuile val")
    parser.add_argument("--threshold", type=float, default=None, help="Optionnel: filtre |s| < threshold")
    args = parser.parse_args()

    train_tile = pd.read_parquet(args.train_tile)
    val_full = pd.read_parquet(args.val_full)

    required = {"x", "y", "z", "s"}
    for name, df in [("train_tile", train_tile), ("val_full", val_full)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: colonnes manquantes {sorted(missing)}")

    x_min, x_max = train_tile["x"].min(), train_tile["x"].max()
    y_min, y_max = train_tile["y"].min(), train_tile["y"].max()

    mask = (
        (val_full["x"] >= x_min) & (val_full["x"] <= x_max) &
        (val_full["y"] >= y_min) & (val_full["y"] <= y_max)
    )

    val_tile = val_full.loc[mask].copy()

    if args.threshold is not None:
        val_tile = val_tile[val_tile["s"].abs() < args.threshold].copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    val_tile.to_parquet(output, index=False)

    print("Train tile bounds:")
    print(f"x: [{x_min}, {x_max}]")
    print(f"y: [{y_min}, {y_max}]")
    print("Train tile points:", len(train_tile))
    print("Val tile points:", len(val_tile))
    print("Saved to:", output)


if __name__ == "__main__":
    main()