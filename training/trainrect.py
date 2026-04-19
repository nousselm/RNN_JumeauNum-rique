import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.utils.data as data_utils
import yaml

from sampling.rectangle_sampling import sample_rectangle
from datasets.sdf_dataset import SDFDataset
from models.deep_sdf import DeepSDF
from evaluation.grid_eval import reconstruct_mesh

from torch.utils.data import DataLoader


def load_config():

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    return config


def train():

    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = config["dataset"]

    points, sdf = sample_rectangle(
        ds["n_points"],
        ds["width"],
        ds["height"],
        ds["surface_ratio"],
        ds["near_ratio"],
        ds["far_ratio"],
        ds["near_std"]
    )

    dataset = SDFDataset(points, sdf)

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2
    )

    model = DeepSDF(
        hidden_dim=config["model"]["hidden_dim"],
        n_layers=config["model"]["layers"],
        use_fourier=config["model"]["use_fourier"],
        fourier_dim=config["model"]["fourier_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    loss_fn = torch.nn.L1Loss()

    for epoch in range(config["training"]["epochs"]):

        total_loss = 0

        for pts, sdf_gt in loader:

            pts = pts.to(device)
            sdf_gt = sdf_gt.to(device)

            pred = model(pts)

            loss = loss_fn(pred, sdf_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch", epoch, "Loss", total_loss / len(loader))

    torch.save(model.state_dict(), "model_rectangle.pth")


    mesh = reconstruct_mesh(
        model,
        config["evaluation"]["grid_res"],
        device
)

    print("Mesh saved : rectangle_reconstruction.obj")
  


if __name__ == "__main__":

    train()