from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_sdf(parquet_path: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["s"])
    sdf = df["s"].to_numpy(dtype=np.float64)
    return sdf


def compute_distribution(sdf: np.ndarray, eps: float) -> dict:
    surface_mask = np.abs(sdf) <= eps
    inside_mask = sdf < -eps
    outside_mask = sdf > eps

    n = len(sdf)
    n_surface = int(surface_mask.sum())
    n_inside = int(inside_mask.sum())
    n_outside = int(outside_mask.sum())

    return {
        "eps": eps,
        "total": n,
        "surface_count": n_surface,
        "inside_count": n_inside,
        "outside_count": n_outside,
        "surface_pct": 100.0 * n_surface / n,
        "inside_pct": 100.0 * n_inside / n,
        "outside_pct": 100.0 * n_outside / n,
    }


def print_distribution(stats: dict) -> None:
    print(f"\n=== Distribution pour eps = {stats['eps']:.8f} ===")
    print(f"Total     : {stats['total']}")
    print(f"Surface   : {stats['surface_count']} ({stats['surface_pct']:.2f}%)")
    print(f"Intérieur : {stats['inside_count']} ({stats['inside_pct']:.2f}%)")
    print(f"Extérieur : {stats['outside_count']} ({stats['outside_pct']:.2f}%)")


def search_best_eps(
    sdf: np.ndarray,
    eps_min: float,
    eps_max: float,
    num_steps: int,
    target_surface_pct: float = 70.0,
    target_inside_pct: float = 15.0,
    target_outside_pct: float = 15.0,
) -> dict:
    best = None
    best_score = float("inf")

    eps_values = np.linspace(eps_min, eps_max, num_steps)

    for eps in eps_values:
        stats = compute_distribution(sdf, float(eps))

        score = (
            abs(stats["surface_pct"] - target_surface_pct)
            + abs(stats["inside_pct"] - target_inside_pct)
            + abs(stats["outside_pct"] - target_outside_pct)
        )

        if score < best_score:
            best_score = score
            best = stats.copy()
            best["score"] = score

    return best


def main():
    parser = argparse.ArgumentParser(description="Analyse la distribution SDF et cherche un eps optimal.")
    parser.add_argument("--input", required=True, help="Chemin du fichier parquet")
    parser.add_argument("--eps", type=float, default=None, help="Seuil eps pour compter surface/intérieur/extérieur")
    parser.add_argument("--search", action="store_true", help="Cherche un eps optimal")
    parser.add_argument("--eps-min", type=float, default=1e-6, help="Valeur minimale de eps pour la recherche")
    parser.add_argument("--eps-max", type=float, default=0.05, help="Valeur maximale de eps pour la recherche")
    parser.add_argument("--num-steps", type=int, default=200, help="Nombre de valeurs testées entre eps-min et eps-max")
    parser.add_argument("--target-surface", type=float, default=70.0, help="Pourcentage cible surface")
    parser.add_argument("--target-inside", type=float, default=15.0, help="Pourcentage cible intérieur")
    parser.add_argument("--target-outside", type=float, default=15.0, help="Pourcentage cible extérieur")
    args = parser.parse_args()

    sdf = load_sdf(args.input)

    print(f"Fichier : {args.input}")
    print(f"Nombre de points : {len(sdf)}")
    print(f"SDF min : {sdf.min():.8f}")
    print(f"SDF max : {sdf.max():.8f}")
    print(f"SDF mean: {sdf.mean():.8f}")

    if args.eps is not None:
        stats = compute_distribution(sdf, args.eps)
        print_distribution(stats)

    if args.search:
        best = search_best_eps(
            sdf=sdf,
            eps_min=args.eps_min,
            eps_max=args.eps_max,
            num_steps=args.num_steps,
            target_surface_pct=args.target_surface,
            target_inside_pct=args.target_inside,
            target_outside_pct=args.target_outside,
        )

        print("\n=== Meilleur eps trouvé ===")
        print_distribution(best)
        print(f"Score d'écart total : {best['score']:.4f}")


if __name__ == "__main__":
    main()