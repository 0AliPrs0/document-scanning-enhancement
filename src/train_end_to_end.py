import os
import sys
import math
import json
import random
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import kornia

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_enhancement import DocumentEnhancementUNet
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


def extract_coords_differentiable(heatmaps, temperature=20.0):
    B, C, H, W = heatmaps.shape
    eps = 1e-6
    scores = torch.log(heatmaps.clamp(min=eps))
    scores = scores.view(B, C, -1)
    probs = torch.softmax(scores * temperature, dim=2)
    probs = probs.view(B, C, H, W)
    y_grid = torch.linspace(0.0, 1.0, H, device=heatmaps.device).view(1, 1, H, 1)
    x_grid = torch.linspace(0.0, 1.0, W, device=heatmaps.device).view(1, 1, 1, W)
    x_coords = torch.sum(probs * x_grid, dim=(2, 3))
    y_coords = torch.sum(probs * y_grid, dim=(2, 3))
    return torch.stack([x_coords, y_coords], dim=2)


def polygon_area(points):
    x = points[..., 0]
    y = points[..., 1]
    return 0.5 * torch.abs(
        x[:, 0] * y[:, 1]
        + x[:, 1] * y[:, 2]
        + x[:, 2] * y[:, 3]
        + x[:, 3] * y[:, 0]
        - y[:, 0] * x[:, 1]
        - y[:, 1] * x[:, 2]
        - y[:, 2] * x[:, 3]
        - y[:, 3] * x[:, 0]
    )


def cross_z(a, b, c):
    ab = b - a
    ac = c - a
    return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]


def get_safe_corners(corners, width, height):
    tl = corners[:, 0]
    tr = corners[:, 1]
    br = corners[:, 2]
    bl = corners[:, 3]
    top = torch.linalg.vector_norm(tr - tl, dim=1)
    right = torch.linalg.vector_norm(br - tr, dim=1)
    bottom = torch.linalg.vector_norm(br - bl, dim=1)
    left = torch.linalg.vector_norm(bl - tl, dim=1)
    area = polygon_area(corners)
    c1 = cross_z(tl, tr, br)
    c2 = cross_z(tr, br, bl)
    c3 = cross_z(br, bl, tl)
    c4 = cross_z(bl, tl, tr)
    same_sign = ((c1 > 0) & (c2 > 0) & (c3 > 0) & (c4 > 0)) | (
        (c1 < 0) & (c2 < 0) & (c3 < 0) & (c4 < 0)
    )
    min_side = min(width, height)
    min_side_length = min_side * 0.02
    min_area = width * height * 0.01
    valid = (
        torch.isfinite(corners).all(dim=(1, 2))
        & (top > min_side_length)
        & (right > min_side_length)
        & (bottom > min_side_length)
        & (left > min_side_length)
        & (area > min_area)
        & same_sign
    )
    clamped = torch.clamp(corners, min=0.0, max=max(width - 1.0, height - 1.0))
    x = torch.clamp(clamped[..., 0], 0.0, width - 1.0)
    y = torch.clamp(clamped[..., 1], 0.0, height - 1.0)
    clamped = torch.stack([x, y], dim=-1)
    fallback = (
        torch.tensor(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            device=corners.device,
            dtype=corners.dtype,
        )
        .unsqueeze(0)
        .expand(corners.shape[0], -1, -1)
    )
    valid = valid.view(-1, 1, 1)
    return torch.where(valid, clamped, fallback)


class EndToEndScanner(nn.Module):
    def __init__(self, corner_model, enhancement_model, target_size=(512, 512)):
        super().__init__()
        self.corner_model = corner_model
        self.enhancement_model = enhancement_model
        self.target_size = target_size
        w, h = target_size
        self.register_buffer(
            "dest_pts",
            torch.tensor(
                [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                dtype=torch.float32,
            ),
        )

    def forward(self, raw_photo):
        B, C, H, W = raw_photo.shape
        corner_heatmaps = self.corner_model(raw_photo)
        norm_corners = extract_coords_differentiable(corner_heatmaps, temperature=20.0)
        scale = torch.tensor(
            [W - 1.0, H - 1.0], device=norm_corners.device, dtype=norm_corners.dtype
        ).view(1, 1, 2)
        predicted_corners = norm_corners * scale
        predicted_corners = get_safe_corners(predicted_corners, W, H)
        dest_pts_batch = (
            self.dest_pts.to(device=raw_photo.device, dtype=raw_photo.dtype)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        src_pts = predicted_corners.to(dtype=raw_photo.dtype)
        H_mat = kornia.geometry.transform.get_perspective_transform(
            src_pts, dest_pts_batch
        )
        rectified_crops = kornia.geometry.transform.warp_perspective(
            raw_photo, H_mat, self.target_size
        )
        enhanced_out = self.enhancement_model(rectified_crops)
        return enhanced_out, rectified_crops, corner_heatmaps, norm_corners


class E2EDataset(Dataset):
    def __init__(self, data_dir, heatmap_size=256, sigma=7.0):
        self.raw_dir = os.path.join(data_dir, "raw_inputs")
        self.target_dir = os.path.join(data_dir, "clean_targets")
        self.labels_path = os.path.join(data_dir, "corners.json")
        if not os.path.exists(self.labels_path):
            raise RuntimeError(f"Missing {self.labels_path}")
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)
        self.files = sorted(
            [
                f
                for f in self.labels.keys()
                if os.path.exists(os.path.join(self.raw_dir, f))
                and os.path.exists(os.path.join(self.target_dir, f))
            ]
        )
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        if not self.files:
            raise RuntimeError(f"No valid E2E samples in {data_dir}")

    def __len__(self):
        return len(self.files)

    def make_heatmaps(self, corners):
        size = self.heatmap_size
        heatmaps = np.zeros((4, size, size), dtype=np.float32)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        for i in range(4):
            x = float(corners[i, 0]) * (size - 1)
            y = float(corners[i, 1]) * (size - 1)
            dist_sq = (xx - x) ** 2 + (yy - y) ** 2
            heatmaps[i] = np.exp(-dist_sq / (2.0 * self.sigma * self.sigma))
        return heatmaps

    def __getitem__(self, idx):
        filename = self.files[idx]
        raw = cv2.imread(os.path.join(self.raw_dir, filename))
        target = cv2.imread(os.path.join(self.target_dir, filename))
        if raw is None or target is None:
            raise RuntimeError(f"Could not read {filename}")
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        corners = np.array(self.labels[filename], dtype=np.float32)
        heatmaps = self.make_heatmaps(corners)
        raw_tensor = torch.from_numpy(raw.transpose(2, 0, 1)).float() / 255.0
        target_tensor = torch.from_numpy(target.transpose(2, 0, 1)).float() / 255.0
        corners_tensor = torch.from_numpy(corners).float()
        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        return {
            "raw_photo": raw_tensor,
            "clean_target": target_tensor,
            "corners": corners_tensor,
            "heatmaps": heatmaps_tensor,
        }


class EnhancementLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def ssim_loss(self, pred, target):
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = F.avg_pool2d(pred, 11, 1, 5)
        mu_y = F.avg_pool2d(target, 11, 1, 5)
        sigma_x = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, 11, 1, 5) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim = numerator / (denominator + 1e-6)
        return 1.0 - ssim.mean()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        edge = F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
        return 0.50 * l1 + 0.30 * ssim + 0.20 * edge


class WeightedHeatmapLoss(nn.Module):
    def __init__(self, positive_weight=10.0):
        super().__init__()
        self.positive_weight = positive_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weights = 1.0 + self.positive_weight * target
        return (bce * weights).mean()


def load_model_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return model


def train_end_to_end(
    model,
    train_loader,
    val_loader,
    epochs,
    corner_lr,
    enhancement_lr,
    save_path,
    device,
):
    enhancement_criterion = EnhancementLoss().to(device)
    heatmap_criterion = WeightedHeatmapLoss(positive_weight=10.0).to(device)
    optimizer = optim.AdamW(
        [
            {"params": model.corner_model.parameters(), "lr": corner_lr},
            {"params": model.enhancement_model.parameters(), "lr": enhancement_lr},
        ],
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_val_loss = float("inf")
    train_history, val_history = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"E2E Epoch [{epoch + 1}/{epochs}]")

        for batch in loop:
            raw = batch["raw_photo"].to(device, non_blocking=True)
            target = batch["clean_target"].to(device, non_blocking=True)
            target_heatmaps = batch["heatmaps"].to(device, non_blocking=True)
            target_corners = batch["corners"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            enhanced, rectified, corner_logits, predicted_corners = model(raw)

            enhancement_loss = enhancement_criterion(enhanced, target)
            heatmap_loss = heatmap_criterion(corner_logits, target_heatmaps)
            coordinate_loss = F.smooth_l1_loss(predicted_corners, target_corners)
            loss = (
                0.65 * enhancement_loss + 0.20 * heatmap_loss + 0.15 * coordinate_loss
            )

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                enh=f"{enhancement_loss.item():.4f}",
                corner=f"{coordinate_loss.item():.4f}",
            )

        train_loss = running_loss / max(len(train_loader), 1)
        train_history.append(train_loss)

        model.eval()
        val_loss_total = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                raw = batch["raw_photo"].to(device, non_blocking=True)
                target = batch["clean_target"].to(device, non_blocking=True)
                target_heatmaps = batch["heatmaps"].to(device, non_blocking=True)
                target_corners = batch["corners"].to(device, non_blocking=True)
                enhanced, rectified, corner_logits, predicted_corners = model(raw)
                enhancement_loss = enhancement_criterion(enhanced, target)
                heatmap_loss = heatmap_criterion(corner_logits, target_heatmaps)
                coordinate_loss = F.smooth_l1_loss(predicted_corners, target_corners)
                loss = (
                    0.65 * enhancement_loss
                    + 0.20 * heatmap_loss
                    + 0.15 * coordinate_loss
                )
                if torch.isfinite(loss):
                    val_loss_total += loss.item() * raw.size(0)
                    val_count += raw.size(0)

        val_loss = val_loss_total / max(val_count, 1) if val_count > 0 else float("inf")
        val_history.append(val_loss)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: train={train_loss:.6f} val={val_loss:.6f} lr_corner={optimizer.param_groups[0]['lr']:.2e} lr_enh={optimizer.param_groups[1]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )
            print(f" -> Best model saved at Epoch {epoch + 1}")

    curve_path = save_path.replace(".pth", "_loss_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_history, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("E2E Loss")
    plt.title("End-to-End Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--corner_weights", type=str, required=True)
    parser.add_argument("--enh_weights", type=str, required=True)
    parser.add_argument("--corner_dropout", action="store_true")
    parser.add_argument("--enh_dropout", action="store_true")
    parser.add_argument("--corner_lr", type=float, default=1e-5)
    parser.add_argument("--enh_lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = E2EDataset("data/e2e_dataset/train", heatmap_size=256, sigma=7.0)
    val_dataset = E2EDataset("data/e2e_dataset/valid", heatmap_size=256, sigma=7.0)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    corner_model = CornerHeatmapUNet(use_dropout=args.corner_dropout)
    enhancement_model = DocumentEnhancementUNet(use_dropout=args.enh_dropout)

    corner_model = load_model_weights(corner_model, args.corner_weights, "cpu")
    enhancement_model = load_model_weights(enhancement_model, args.enh_weights, "cpu")

    e2e_model = EndToEndScanner(corner_model, enhancement_model).to(device)

    save_path = "output/best_e2e_model.pth"
    train_end_to_end(
        model=e2e_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        corner_lr=args.corner_lr,
        enhancement_lr=args.enh_lr,
        save_path=save_path,
        device=device,
    )


if __name__ == "__main__":
    main()
