# from __future__ import annotations

# import os
# import numpy as np
# import torch
# import trimesh
# from skimage import measure


# def denormalize_with_center_scale(values: np.ndarray, center: float, scale: float) -> np.ndarray:
#     values = np.asarray(values, dtype=np.float64)
#     return values * scale + center


# def reconstruct_mesh(
#     model,
#     res_x: int = 128,
#     res_y: int = 128,
#     res_z: int = 128,
#     device: str | torch.device = "cpu",
#     output_path: str = "reconstruction.obj",
#     normalization: dict | None = None,
# ):
#     model.eval()

#     if normalization is None:
#         raise ValueError("Le dictionnaire de normalisation est requis.")

#     x_norm_min = float(normalization["x_norm_min"])
#     x_norm_max = float(normalization["x_norm_max"])
#     y_norm_min = float(normalization["y_norm_min"])
#     y_norm_max = float(normalization["y_norm_max"])
#     z_norm_min = float(normalization["z_norm_min"])
#     z_norm_max = float(normalization["z_norm_max"])

#     xs = torch.linspace(x_norm_min, x_norm_max, res_x, dtype=torch.float32)
#     ys = torch.linspace(y_norm_min, y_norm_max, res_y, dtype=torch.float32)
#     zs = torch.linspace(z_norm_min, z_norm_max, res_z, dtype=torch.float32)

#     grid = torch.stack(
#         torch.meshgrid(xs, ys, zs, indexing="ij"),
#         dim=-1
#     ).reshape(-1, 3)

#     grid = grid.to(device)

#     with torch.no_grad():
#         sdf = model(grid).squeeze(-1).cpu().numpy()

#     sdf = sdf.reshape(res_x, res_y, res_z)

#     sdf_min = float(sdf.min())
#     sdf_max = float(sdf.max())

#     print("SDF min:", sdf_min)
#     print("SDF max:", sdf_max)

#     if not np.isfinite(sdf).all():
#         raise ValueError("Le volume SDF contient des NaN ou des inf.")

#     if not (sdf_min <= 0.0 <= sdf_max):
#         level = 0.5 * (sdf_min + sdf_max)
#         print(f"0.0 n'est pas dans [min, max], utilisation de level={level}")
#     else:
#         level = 0.0

#     verts, faces, _, _ = measure.marching_cubes(sdf, level=level)

#     # Conversion indices voxel -> coordonnées normalisées
#     scale_x = (x_norm_max - x_norm_min) / (res_x - 1)
#     scale_y = (y_norm_max - y_norm_min) / (res_y - 1)
#     scale_z = (z_norm_max - z_norm_min) / (res_z - 1)

#     verts[:, 0] = verts[:, 0] * scale_x + x_norm_min
#     verts[:, 1] = verts[:, 1] * scale_y + y_norm_min
#     verts[:, 2] = verts[:, 2] * scale_z + z_norm_min

#     # Dénormalisation si on a centre + échelle
#     x_center = normalization.get("x_center", None)
#     x_scale = normalization.get("x_scale", None)
#     y_center = normalization.get("y_center", None)
#     y_scale = normalization.get("y_scale", None)
#     z_center = normalization.get("z_center", None)
#     z_scale = normalization.get("z_scale", None)

#     if x_center is not None and x_scale is not None:
#         verts[:, 0] = denormalize_with_center_scale(verts[:, 0], float(x_center), float(x_scale))

#     if y_center is not None and y_scale is not None:
#         verts[:, 1] = denormalize_with_center_scale(verts[:, 1], float(y_center), float(y_scale))

#     if z_center is not None and z_scale is not None:
#         # verts[:, 2] = denormalize_with_center_scale(verts[:, 2], float(z_center), float(z_scale))
#         pass

    
#     #ajout ici essai 3 
#     print("Verts min:", verts.min(axis=0))
#     print("Verts max:", verts.max(axis=0))
#     #fin ajout essai 3
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

#     # # On garde la plus grosse composante connexe
#     # components = mesh.split(only_watertight=False)
#     # if len(components) > 0:
#     #     mesh = max(components, key=lambda m: len(m.faces))
#     print("Nombre de composantes avant filtrage :", len(mesh.split(only_watertight=False))) #ajout essai 3

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

# #ajout essai 3
#     print("Mesh bounds:", mesh.bounds)
#     print("Nb vertices:", len(mesh.vertices))
#     print("Nb faces:", len(mesh.faces))
# #fin ajout essai 3
#     mesh.export(output_path)

#     print("Vertices:", len(mesh.vertices))
#     print("Faces:", len(mesh.faces))
#     print("Bounds:", mesh.bounds)
#     print(f"Reconstruction saved to {output_path}")

#     return mesh, verts, faces

from __future__ import annotations

import os
import numpy as np
import torch
import trimesh
from skimage import measure


def reconstruct_mesh(
    model,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    res_x: int = 192,
    res_y: int = 192,
    res_z: int = 192,
    device: str | torch.device = "cpu",
    output_path: str = "reconstruction.obj",
):
    model.eval()

    xs = torch.linspace(x_min, x_max, res_x, dtype=torch.float32)
    ys = torch.linspace(y_min, y_max, res_y, dtype=torch.float32)
    zs = torch.linspace(z_min, z_max, res_z, dtype=torch.float32)

    grid = torch.stack(
        torch.meshgrid(xs, ys, zs, indexing="ij"),
        dim=-1
    ).reshape(-1, 3)

    grid = grid.to(device)

    with torch.no_grad():
        batch_size = 50000  # ajuste si besoin
        sdf_list = []

    for i in range(0, grid.shape[0], batch_size):
        batch = grid[i:i+batch_size]
        pred = model(batch).squeeze(-1).cpu()
        sdf_list.append(pred)

    sdf = torch.cat(sdf_list, dim=0).detach().numpy()

    sdf = sdf.reshape(res_x, res_y, res_z)

    sdf_min = float(sdf.min())
    sdf_max = float(sdf.max())

    print("SDF min:", sdf_min)
    print("SDF max:", sdf_max)

    if not np.isfinite(sdf).all():
        raise ValueError("Le volume SDF contient des NaN ou des inf.")

    if not (sdf_min <= 0.0 <= sdf_max):
        level = 0.5 * (sdf_min + sdf_max)
        print(f"0.0 n'est pas dans [min, max], utilisation de level={level}")
    else:
        level = 0.0

    verts, faces, _, _ = measure.marching_cubes(sdf, level=level)

    scale_x = (x_max - x_min) / (res_x - 1)
    scale_y = (y_max - y_min) / (res_y - 1)
    scale_z = (z_max - z_min) / (res_z - 1)

    verts[:, 0] = verts[:, 0] * scale_x + x_min
    verts[:, 1] = verts[:, 1] * scale_y + y_min
    verts[:, 2] = verts[:, 2] * scale_z + z_min

    print("Verts min:", verts.min(axis=0))
    print("Verts max:", verts.max(axis=0))

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    components = mesh.split(only_watertight=False)

    print("Nombre de composantes avant filtrage :", len(components))

    if len(components) > 0:
        mesh = max(components, key=lambda m: len(m.faces))

    print("Mesh bounds:", mesh.bounds)
    print("Nb vertices:", len(mesh.vertices))
    print("Nb faces:", len(mesh.faces))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mesh.export(output_path)
    print(f"Reconstruction saved to {output_path}")

    return mesh, mesh.vertices, mesh.faces
    #return mesh, verts, faces