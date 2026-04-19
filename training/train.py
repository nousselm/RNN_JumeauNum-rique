# from __future__ import annotations

# import os
# import sys
# from pathlib import Path

# import torch
# import yaml
# from torch.utils.data import DataLoader

# # Ajout du dossier racine au PYTHONPATH
# ROOT_DIR = Path(__file__).resolve().parents[1]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.append(str(ROOT_DIR))

# from datasets.airport_dataset import AirportSDFParquetDataset
# from evaluation.grid_eval import reconstruct_mesh
# from models.deep_sdf import DeepSDF


# def load_config(config_path: Path):
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def evaluate_loss(model, loader, loss_fn, device):
#     model.eval()
#     total_loss = 0.0

#     with torch.no_grad():
#         for pts, sdf_gt in loader:
#             pts = pts.to(device)
#             sdf_gt = sdf_gt.to(device)

#             pred = model(pts)
#             loss = loss_fn(pred, sdf_gt)
#             total_loss += loss.item()

#     return total_loss / max(len(loader), 1)


# def train():
#     config_path = ROOT_DIR / "config" / "config.yaml"
#     config = load_config(config_path)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     train_dataset = AirportSDFParquetDataset(
#         parquet_path=str(ROOT_DIR / config["dataset"]["train_parquet"]),
#         clamp=config["dataset"]["clamp"],
#         verbose=config["dataset"].get("verbose", False),
#     )

#     val_dataset = AirportSDFParquetDataset(
#         parquet_path=str(ROOT_DIR / config["dataset"]["val_parquet"]),
#         clamp=config["dataset"]["clamp"],
#         verbose=False,
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=True,
#         num_workers=0,
#         pin_memory=torch.cuda.is_available(),
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config["training"]["batch_size"],
#         shuffle=False,
#         num_workers=0,
#         pin_memory=torch.cuda.is_available(),
#     )

#     model = DeepSDF(
#         hidden_dim=config["model"]["hidden_dim"],
#         n_layers=config["model"]["layers"],
#         use_fourier=config["model"]["use_fourier"],
#         fourier_dim=config["model"]["fourier_dim"],
#         fourier_scale=config["model"]["fourier_scale"],
#     ).to(device)

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=config["training"]["lr"]
#     )

#     loss_fn = torch.nn.L1Loss()

#     output_path = ROOT_DIR / config["evaluation"]["output_path"]
#     output_dir = output_path.parent
#     os.makedirs(output_dir, exist_ok=True)

#     best_model_path = output_dir / "best_model.pth"

#     best_val_loss = float("inf")
#     epochs_without_improvement = 0

#     patience = config["training"]["early_stopping_patience"]
#     min_delta = config["training"]["early_stopping_min_delta"]

#     for epoch in range(config["training"]["epochs"]):
#         model.train()
#         total_train_loss = 0.0

#         for pts, sdf_gt in train_loader:
#             pts = pts.to(device)
#             sdf_gt = sdf_gt.to(device)

#             pred = model(pts)
#             loss = loss_fn(pred, sdf_gt)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_train_loss += loss.item()

#         avg_train_loss = total_train_loss / max(len(train_loader), 1)
#         avg_val_loss = evaluate_loss(model, val_loader, loss_fn, device)

#         print(
#             f"Epoch {epoch + 1}/{config['training']['epochs']} | "
#             f"train_loss={avg_train_loss:.8f} | val_loss={avg_val_loss:.8f}"
#         )

#         if avg_val_loss < best_val_loss - min_delta:
#             best_val_loss = avg_val_loss
#             epochs_without_improvement = 0
#             torch.save(model.state_dict(), best_model_path)
#         else:
#             epochs_without_improvement += 1

#         if epochs_without_improvement >= patience:
#             print(f"Early stopping at epoch {epoch + 1}")
#             break

#     print("Best validation loss:", best_val_loss)

#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     model.to(device)

#     reconstruct_mesh(
#         model=model,
#         res_x=config["evaluation"]["res_x"],
#         res_y=config["evaluation"]["res_y"],
#         res_z=config["evaluation"]["res_z"],
#         device=device,
#         output_path=str(output_path),
#         normalization=config["normalization"],
#     )


# if __name__ == "__main__":
#     train()

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from datasets.airport_dataset import AirportSDFParquetDataset
from evaluation.grid_eval import reconstruct_mesh
from models.deep_sdf import DeepSDF


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for pts, sdf_gt in loader:
            pts = pts.to(device)
            sdf_gt = sdf_gt.to(device)

            pred = model(pts)
            loss = loss_fn(pred, sdf_gt)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def get_bounds_from_dataset(dataset):
    pts = dataset.points.numpy()
    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)
    return x_min, x_max, y_min, y_max, z_min, z_max


def train():
    config_path = ROOT_DIR / "config" / "config.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = AirportSDFParquetDataset(
        parquet_path=str(ROOT_DIR / config["dataset"]["train_parquet"]),
        clamp=config["dataset"]["clamp"],
        verbose=config["dataset"].get("verbose", False),
    )

    val_dataset = AirportSDFParquetDataset(
        parquet_path=str(ROOT_DIR / config["dataset"]["val_parquet"]),
        clamp=config["dataset"]["clamp"],
        verbose=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = DeepSDF(
        hidden_dim=config["model"]["hidden_dim"],
        n_layers=config["model"]["layers"],
        use_fourier=config["model"]["use_fourier"],
        fourier_dim=config["model"]["fourier_dim"],
        fourier_scale=config["model"]["fourier_scale"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]
    )

    loss_fn = torch.nn.L1Loss()

    output_path = ROOT_DIR / config["evaluation"]["output_path"]
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = output_dir / "best_model.pth"

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    patience = config["training"]["early_stopping_patience"]
    min_delta = config["training"]["early_stopping_min_delta"]

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_train_loss = 0.0

        for pts, sdf_gt in train_loader:
            pts = pts.to(device)
            sdf_gt = sdf_gt.to(device)

            pred = model(pts)
            loss = loss_fn(pred, sdf_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']} | "
            f"train_loss={avg_train_loss:.8f} | val_loss={avg_val_loss:.8f}"
        )

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Best validation loss:", best_val_loss)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    x_min, x_max, y_min, y_max, z_min, z_max = get_bounds_from_dataset(train_dataset)

    print("TRAIN TILE BOUNDS:")
    print("x:", x_min, x_max)
    print("y:", y_min, y_max)
    print("z:", z_min, z_max)

    del optimizer
    torch.cuda.empty_cache()

    reconstruct_mesh(
        model=model,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        z_min=float(z_min),
        z_max=float(z_max),
        res_x=config["evaluation"]["res_x"],
        res_y=config["evaluation"]["res_y"],
        res_z=config["evaluation"]["res_z"],
        device=device,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    train()