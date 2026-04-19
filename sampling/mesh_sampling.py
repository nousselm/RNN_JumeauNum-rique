import trimesh
import torch


def sample_mesh(mesh_path, n_points):

    mesh = trimesh.load(mesh_path)

    surface_points, _ = trimesh.sample.sample_surface(mesh, n_points)

    surface_points = torch.tensor(surface_points).float()

    noise = torch.randn_like(surface_points) * 0.02

    near_points = surface_points + noise

    far_points = torch.rand(n_points,3)*4 - 2

    points = torch.cat([surface_points, near_points, far_points],0)

    sdf = torch.tensor(trimesh.proximity.signed_distance(mesh, points.numpy())).float()

    return points, sdf