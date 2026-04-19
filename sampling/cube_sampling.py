import torch

# cube sampling 1er version
# def sdf_cube(points, size=0.5):
#     q = torch.abs(points) - size
#     outside = torch.clamp(q, min=0)
#     inside = torch.minimum(torch.max(q, dim=1).values, torch.tensor(0.0))
#     return torch.norm(outside, dim=1) + inside


# def sample_cube(n_points, cube_size):

#     points = torch.rand(n_points, 3) * 2 - 1

#     sdf = sdf_cube(points, cube_size)

#     return points, sdf

#////////////// V2 cube sampling corrigé pour avoir plus de points sur la surface et autour de la surface

# import torch


# def sdf_cube(points, size=0.5):

#     q = torch.abs(points) - size
#     outside = torch.clamp(q, min=0)
#     inside = torch.minimum(torch.max(q, dim=1).values, torch.tensor(0.0))

#     return torch.norm(outside, dim=1) + inside


# def sample_cube(n_points, cube_size, surface_ratio, near_ratio, far_ratio, near_std):

#     n_surface = int(n_points * surface_ratio)
#     n_near = int(n_points * near_ratio)
#     n_far = int(n_points * far_ratio)

#     # points sur surface
#     surface = torch.rand(n_surface,3) * cube_size * 2 - cube_size

#     face = torch.randint(0,3,(n_surface,))
#     sign = torch.randint(0,2,(n_surface,))*2-1

#     surface[torch.arange(n_surface),face] = sign * cube_size

#     # points proches surface
#     near = surface + torch.randn_like(surface)*near_std

#     # points loin
#     far = torch.rand(n_far,3)*2-1

#     points = torch.cat([surface, near, far], dim=0)

#     sdf = sdf_cube(points, cube_size)

#     return points, sdf

# ///////////////V3 cube sampling corrigé pour limité le bruit

import torch


def sdf_cube(points, size):

    q = torch.abs(points) - size
    outside = torch.clamp(q, min=0)
    inside = torch.minimum(torch.max(q, dim=1).values, torch.tensor(0.0))

    return torch.norm(outside, dim=1) + inside


def sample_cube(n_points, cube_size, surface_ratio, near_ratio, far_ratio, near_std):

    n_surface = int(n_points * surface_ratio)
    n_near = int(n_points * near_ratio)
    n_far = int(n_points * far_ratio)

    # surface sampling correct
    faces = torch.randint(0,6,(n_surface,))
    surface = torch.rand(n_surface,3)*2*cube_size - cube_size

    for i,f in enumerate(faces):

        if f==0: surface[i,0] = cube_size
        if f==1: surface[i,0] = -cube_size
        if f==2: surface[i,1] = cube_size
        if f==3: surface[i,1] = -cube_size
        if f==4: surface[i,2] = cube_size
        if f==5: surface[i,2] = -cube_size

    # near surface
    near = surface + torch.randn_like(surface)*near_std

    # far
    far = torch.rand(n_far,3)*2 - 1

    points = torch.cat([surface, near, far],0)

    sdf = sdf_cube(points, cube_size)

    return points, sdf

