import os
import sys
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_corners import HeatmapCornerDataset
from model_corners import CornerHeatmapUNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightedHeatmapBCELoss(nn.Module):
    def __init__(self, positive_weight=10.0):
        super().__init__()
        self.positive_weight = positive_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        weights = 1.0 + self.positive_weight * targets

        return torch.mean(weights * bce)


def extract_coords_from_heatmaps(
    logits,
    temperature=10.0,
):
    batch_size, channels, height, width = logits.shape

    flat_logits = logits.reshape(
        batch_size,
        channels,
        -1,
    )

    probabilities = torch.softmax(
        flat_logits * temperature,
        dim=2,
    )

    y_grid = torch.linspace(
        0.0,
        1.0,
        height,
        device=logits.device,
    ).view(1, 1, height, 1)

    x_grid = torch.linspace(
        0.0,
        1.0,
        width,
        device=logits.device,
    ).view(1, 1, 1, width)

    y_grid = y_grid.expand(
        batch_size,
        channels,
        height,
        width,
    )

    x_grid = x_grid.expand(
        batch_size,
        channels,
        height,
        width,
    )

    probabilities = probabilities.reshape(
        batch_size,
        channels,
        height,
        width,
    )

    x_coords = torch.sum(
        probabilities * x_grid,
        dim=(2, 3),
    )

    y_coords = torch.sum(
        probabilities * y_grid,
        dim=(2, 3),
    )

    return torch.stack(
        [x_coords, y_coords],
        dim=2,
    )


def calculate_euclidean_error(
    predictions,
    targets,
    img_size,
):
    distances = torch.norm(
        predictions - targets,
        dim=2,
    )

    return (distances * img_size).mean().item()


def evaluate_model(
    model,
    dataloader,
    device,
    img_size,
):
    model.eval()

    total_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["corners"].to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            predictions = extract_coords_from_heatmaps(logits)

            batch_error = calculate_euclidean_error(
                predictions,
                targets,
                img_size,
            )

            batch_size = images.size(0)

            total_error += batch_error * batch_size

            total_samples += batch_size

    return total_error / max(
        total_samples,
        1,
    )


def evaluate_heatmap_loss(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["heatmaps"].to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = criterion(
                logits,
                targets,
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            total_samples += batch_size

    return total_loss / max(
        total_samples,
        1,
    )


def train_heatmap(
    model,
    train_loader,
    val_loader,
    test_loader,
    img_size,
    epochs,
    lr,
    save_path,
    report_path,
    use_dropout=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = WeightedHeatmapBCELoss(positive_weight=10.0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()

        running_train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{epochs}]",
        )

        for batch in progress:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["heatmaps"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_train_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = running_train_loss / max(len(train_loader), 1)

        train_losses.append(train_loss)

        val_loss = evaluate_heatmap_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        val_losses.append(val_loss)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                save_path,
            )

    title_suffix = "with Dropout" if use_dropout else "without Dropout"

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, epochs + 1),
        train_losses,
        label="Train Loss",
    )

    plt.plot(
        range(1, epochs + 1),
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.title(f"Heatmap Training and Validation Loss ({title_suffix})")

    plt.legend()
    plt.grid(True)

    plot_path = save_path.replace(
        ".pth",
        "_loss_curve.png",
    )

    plt.savefig(plot_path)
    plt.close()

    print(f"Loss curve saved to {plot_path}")

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
        )
    )

    train_error = evaluate_model(
        model,
        train_loader,
        device,
        img_size,
    )

    val_error = evaluate_model(
        model,
        val_loader,
        device,
        img_size,
    )

    test_error = evaluate_model(
        model,
        test_loader,
        device,
        img_size,
    )

    metrics = {
        "train_error_px": train_error,
        "val_error_px": val_error,
        "test_error_px": test_error,
        "best_val_heatmap_loss": best_val_loss,
    }

    report_dir = os.path.dirname(report_path)

    if report_dir:
        os.makedirs(
            report_dir,
            exist_ok=True,
        )

    with open(report_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nFinal Results")
    print("=" * 45)
    print(f"Train Error: {train_error:.4f} px")
    print(f"Validation Error: {val_error:.4f} px")
    print(f"Test Error: {test_error:.4f} px")
    print(f"Best Validation Loss: " f"{best_val_loss:.6f}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--heatmap_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=7.0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--use_dropout",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    train_dataset = HeatmapCornerDataset(
        "data/precomputed_corners/train",
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
    )

    val_dataset = HeatmapCornerDataset(
        "data/precomputed_corners/valid",
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
    )

    test_dataset = HeatmapCornerDataset(
        "data/precomputed_corners/test",
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
    )

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
        persistent_workers=True,
    )

    suffix = "dropout" if args.use_dropout else "nodrop"

    save_path = f"/kaggle/working/" f"best_heatmap_{suffix}.pth"

    report_path = f"/kaggle/working/" f"metrics_heatmap_{suffix}.json"

    model = CornerHeatmapUNet(use_dropout=args.use_dropout)

    train_heatmap(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        img_size=args.img_size,
        epochs=args.epochs,
        lr=args.lr,
        save_path=save_path,
        report_path=report_path,
        use_dropout=args.use_dropout,
    )
