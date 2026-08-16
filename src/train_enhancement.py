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
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_enhancement import PrecomputedDocumentDataset
from model_enhancement import DocumentEnhancementUNet
from evaluate import evaluate_model


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)

        ssim_loss = 1.0 - self.ssim(pred, target)

        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)

        pred_edge_x = nn.functional.conv2d(
            pred_gray,
            self.sobel_x,
            padding=1,
        )

        pred_edge_y = nn.functional.conv2d(
            pred_gray,
            self.sobel_y,
            padding=1,
        )

        target_edge_x = nn.functional.conv2d(
            target_gray,
            self.sobel_x,
            padding=1,
        )

        target_edge_y = nn.functional.conv2d(
            target_gray,
            self.sobel_y,
            padding=1,
        )

        edge_loss = self.l1_loss(
            pred_edge_x,
            target_edge_x,
        ) + self.l1_loss(
            pred_edge_y,
            target_edge_y,
        )

        return 0.2 * l1 + 0.4 * edge_loss + 0.4 * ssim_loss


def train_enhancement_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=20,
    lr=1e-4,
    save_path="best_weights.pth",
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

    criterion = CombinedLoss().to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
        )

        for batch in loop:
            inputs = batch["degraded_rectified"].to(device)
            targets = batch["clean_target"].to(device)

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

        avg_train = running_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["degraded_rectified"].to(device)
                targets = batch["clean_target"].to(device)

                outputs = model(inputs)

                val_loss += criterion(
                    outputs,
                    targets,
                ).item()

        avg_val = val_loss / max(len(val_loader), 1)
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
    plt.title(f"Enhancement Training and Validation Loss ({title_suffix})")
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

    print("\nCalculating Final Metrics...")

    t_psnr, t_ssim, _, _ = evaluate_model(
        model,
        train_loader,
        device,
    )

    v_psnr, v_ssim, _, _ = evaluate_model(
        model,
        val_loader,
        device,
    )

    test_psnr, test_ssim, b_psnr_test, b_ssim_test = evaluate_model(
        model,
        test_loader,
        device,
    )

    metrics = {
        "baseline": {
            "psnr": b_psnr_test,
            "ssim": b_ssim_test,
        },
        "train": {
            "psnr": t_psnr,
            "ssim": t_ssim,
        },
        "val": {
            "psnr": v_psnr,
            "ssim": v_ssim,
        },
        "test": {
            "psnr": test_psnr,
            "ssim": test_ssim,
        },
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

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.axis("tight")
    ax.axis("off")

    cell_text = [
        [
            "Baseline",
            f"{b_psnr_test:.4f}",
            f"{b_ssim_test:.4f}",
        ],
        [
            "Training",
            f"{t_psnr:.4f}",
            f"{t_ssim:.4f}",
        ],
        [
            "Validation",
            f"{v_psnr:.4f}",
            f"{v_ssim:.4f}",
        ],
        [
            "Test",
            f"{test_psnr:.4f}",
            f"{test_ssim:.4f}",
        ],
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=["Split", "PSNR", "SSIM"],
        cellLoc="center",
        loc="center",
    )

    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    table_path = save_path.replace(
        ".pth",
        "_metrics_table.png",
    )

    plt.savefig(
        table_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    print(f"Metrics table saved to {table_path}")

    print("\n" + "=" * 45)
    print(f"{'Split':<20} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 45)
    print(f"{'Baseline':<20} | " f"{b_psnr_test:<10.4f} | " f"{b_ssim_test:<10.4f}")
    print(f"{'Training':<20} | " f"{t_psnr:<10.4f} | " f"{t_ssim:<10.4f}")
    print(f"{'Validation':<20} | " f"{v_psnr:<10.4f} | " f"{v_ssim:<10.4f}")
    print(f"{'Test':<20} | " f"{test_psnr:<10.4f} | " f"{test_ssim:<10.4f}")
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
        "--use_dropout",
        action="store_true",
    )

    args = parser.parse_args()

    train_ds = PrecomputedDocumentDataset(
        "data/precomputed_enhancements/train",
        target_size=(
            args.img_size,
            args.img_size,
        ),
    )

    val_ds = PrecomputedDocumentDataset(
        "data/precomputed_enhancements/valid",
        target_size=(
            args.img_size,
            args.img_size,
        ),
    )

    test_ds = PrecomputedDocumentDataset(
        "data/precomputed_enhancements/test",
        target_size=(
            args.img_size,
            args.img_size,
        ),
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
    )

    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
    )

    suffix = "dropout" if args.use_dropout else "nodrop"

    save_path = f"/kaggle/working/" f"best_enhancement_{suffix}.pth"

    report_path = f"/kaggle/working/" f"metrics_enhancement_{suffix}.json"

    model = DocumentEnhancementUNet(use_dropout=args.use_dropout)

    train_enhancement_model(
        model=model,
        train_loader=train_ld,
        val_loader=val_ld,
        test_loader=test_ld,
        num_epochs=args.epochs,
        save_path=save_path,
        report_path=report_path,
        use_dropout=args.use_dropout,
    )
