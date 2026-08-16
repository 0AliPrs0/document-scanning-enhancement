import os
import sys
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_corners import RegressionCornerDataset
from model_corners import CornerDirectRegressor


def calculate_euclidean_error(
    preds,
    targets,
    img_size,
):
    preds = preds.view(-1, 4, 2) * img_size
    targets = targets.view(-1, 4, 2) * img_size

    errors = torch.norm(
        preds - targets,
        dim=2,
    )

    return errors.mean().item()


def evaluate_regression(
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
            inputs = batch["image"].to(device)
            targets = batch["corners"].to(device)

            outputs = model(inputs)

            error = calculate_euclidean_error(
                outputs,
                targets,
                img_size,
            )

            batch_size = inputs.size(0)

            total_error += error * batch_size
            total_samples += batch_size

    return total_error / max(total_samples, 1)


def train_regression(
    model,
    train_loader,
    val_loader,
    test_loader,
    img_size,
    num_epochs=20,
    lr=1e-4,
    save_path="best_reg.pth",
    report_path="report.json",
    use_dropout=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.L1Loss()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}] (Reg)",
        )

        for batch in loop:
            inputs = batch["image"].to(device)
            targets = batch["corners"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss=f"{loss.item():.5f}")

        avg_train = running_loss / max(
            len(train_loader),
            1,
        )

        train_losses.append(avg_train)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                targets = batch["corners"].to(device)

                outputs = model(inputs)

                val_loss += criterion(
                    outputs,
                    targets,
                ).item()

        avg_val = val_loss / max(
            len(val_loader),
            1,
        )

        val_losses.append(avg_val)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={avg_train:.6f} "
            f"val_loss={avg_val:.6f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val

            torch.save(
                model.state_dict(),
                save_path,
            )

    title_suffix = "with Dropout" if use_dropout else "without Dropout"

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, num_epochs + 1),
        train_losses,
        label="Train Loss",
    )

    plt.plot(
        range(1, num_epochs + 1),
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Regression Training and Validation Loss ({title_suffix})")
    plt.legend()
    plt.grid(True)

    plot_path = save_path.replace(
        ".pth",
        "_loss_curve.png",
    )

    plt.savefig(plot_path)
    plt.close()

    print(f"Loss curve saved to {plot_path}")

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
        )
    )

    train_error = evaluate_regression(
        model,
        train_loader,
        device,
        img_size,
    )

    val_error = evaluate_regression(
        model,
        val_loader,
        device,
        img_size,
    )

    test_error = evaluate_regression(
        model,
        test_loader,
        device,
        img_size,
    )

    metrics = {
        "train_error_px": train_error,
        "val_error_px": val_error,
        "test_error_px": test_error,
        "best_val_loss": best_val_loss,
    }

    report_dir = os.path.dirname(report_path)

    if report_dir:
        os.makedirs(
            report_dir,
            exist_ok=True,
        )

    with open(report_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    print("\nFinal Results")
    print("=" * 45)
    print(f"Train Error: {train_error:.4f} px")
    print(f"Validation Error: {val_error:.4f} px")
    print(f"Test Error: {test_error:.4f} px")
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
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--use_dropout",
        action="store_true",
    )

    args = parser.parse_args()

    train_ds = RegressionCornerDataset("data/precomputed_corners/train")

    val_ds = RegressionCornerDataset("data/precomputed_corners/valid")

    test_ds = RegressionCornerDataset("data/precomputed_corners/test")

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    suffix = "dropout" if args.use_dropout else "nodrop"

    save_path = f"/kaggle/working/" f"best_reg_{suffix}.pth"

    report_path = f"/kaggle/working/" f"metrics_reg_{suffix}.json"

    model = CornerDirectRegressor(use_dropout=args.use_dropout)

    train_regression(
        model=model,
        train_loader=train_ld,
        val_loader=val_ld,
        test_loader=test_ld,
        img_size=args.img_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=save_path,
        report_path=report_path,
        use_dropout=args.use_dropout,
    )
