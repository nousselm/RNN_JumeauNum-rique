import torch 

def sdf_rectangle(points, width=1.0, height=0.5):
    q = torch.abs(points[:, :2]) - torch.tensor([width / 2, height / 2])
    outside = torch.clamp(q, min=0)
    inside = torch.minimum(torch.max(q, dim=1).values, torch.tensor(0.0))
    sdf = torch.norm(outside, dim=1) + inside
    return sdf

def sample_rectangle(n_points, width, height, surface_ratio, near_ratio, far_ratio, near_std):
    n_surface = int(n_points * surface_ratio)
    n_near = int(n_points * near_ratio)
    n_far = int(n_points * far_ratio)

    # Points on the surface of the rectangle
    surface = torch.rand(n_surface, 3) * torch.tensor([width, height, 0]) - torch.tensor([width / 2, height / 2, 0])

    axis = torch.randint(0, 2, (n_surface,))  # Randomly select x or y axis
    sign = torch.randint(0, 2, (n_surface,)) * 2 - 1

    surface[torch.arange(n_surface), axis] = sign * torch.tensor([width / 2, height / 2])[axis]

    # Points near the surface
    near = surface + torch.randn_like(surface) * near_std

    # Points far from the surface
    far = torch.rand(n_far, 3) * 2 - 1

    points = torch.cat([surface, near, far], dim=0)

    sdf = sdf_rectangle(points, width, height)

    return points, sdf