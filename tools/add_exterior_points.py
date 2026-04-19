from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--surface-eps", type=float, default=0.01)
    parser.add_argument("--n-samples", type=int, default=200000)
    parser.add_argument("--noise-std", type=float, default=0.02)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    points = df[["x", "y", "z"]].values
    sdf = df["s"].values

    # 1. sélectionner surface
    mask_surface = np.abs(sdf) < args.surface_eps
    surface_points = points[mask_surface]

    print("Surface points:", len(surface_points))

    # 2. sampler des points surface
    idx = np.random.choice(len(surface_points), size=args.n_samples, replace=True)
    base_pts = surface_points[idx]

    # 3. ajouter bruit
    noise = np.random.normal(scale=args.noise_std, size=base_pts.shape)
    new_pts = base_pts + noise

    # 4. sdf = distance signée approx (norme du bruit)
    dist = np.linalg.norm(noise, axis=1)

    # IMPORTANT :
    # moitié extérieur / moitié intérieur
    signs = np.random.choice([-1, 1], size=len(dist))
    new_sdf = dist * signs

    # 5. concat
    all_pts = np.concatenate([points, new_pts], axis=0)
    all_sdf = np.concatenate([sdf, new_sdf], axis=0)

    df_out = pd.DataFrame({
        "x": all_pts[:, 0],
        "y": all_pts[:, 1],
        "z": all_pts[:, 2],
        "s": all_sdf
    })

    df_out.to_parquet(args.output, index=False)

    print("Saved to:", args.output)
    print("New size:", len(df_out))


if __name__ == "__main__":
    main()