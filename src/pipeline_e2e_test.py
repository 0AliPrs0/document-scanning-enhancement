import os
import sys
import cv2
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_enhancement import DocumentEnhancementUNet
from model_corners import CornerHeatmapUNet


def extract_coords(heatmaps):
    B, C, H, W = heatmaps.shape
    flat = heatmaps.view(B, C, -1)
    max_idx = torch.argmax(flat, dim=-1)
    y_coords = (max_idx // W).float() / (H - 1)
    x_coords = (max_idx % W).float() / (W - 1)
    return torch.stack([x_coords, y_coords], dim=2)


def preprocess_image(img, target_size=512):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    x_off = (target_size - nw) // 2
    y_off = (target_size - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas[y_off : y_off + nh, x_off : x_off + nw] = resized
    return canvas, scale, x_off, y_off


def validate_corners(corners, image_shape):
    h, w = image_shape[:2]
    corners = corners.copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    tl, tr, br, bl = corners
    valid = True
    if tl[0] >= tr[0]:
        valid = False
    if bl[0] >= br[0]:
        valid = False
    if tl[1] >= bl[1]:
        valid = False
    if tr[1] >= br[1]:
        valid = False
    return corners, valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--use_dropout", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corner_model = CornerHeatmapUNet(use_dropout=args.use_dropout).to(device)
    enhancement_model = DocumentEnhancementUNet(use_dropout=args.use_dropout).to(device)

    state = torch.load(args.weights, map_location=device)

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    state = {k.replace("module.", ""): v for k, v in state.items()}

    c_state = {
        k[len("corner_model.") :]: v
        for k, v in state.items()
        if k.startswith("corner_model.")
    }
    e_state = {
        k[len("enhancement_model.") :]: v
        for k, v in state.items()
        if k.startswith("enhancement_model.")
    }

    # Fallback: If testing standalone enhancement weights generated directly by train.py
    if not e_state and any(k.startswith("inc.") for k in state.keys()):
        e_state = state

    if not c_state:
        print(
            "WARNING: No corner_model weights found. Ensure you are using an E2E checkpoint for corners."
        )
    else:
        corner_model.load_state_dict(c_state)

    if not e_state:
        raise RuntimeError("No enhancement weights found in checkpoint.")

    enhancement_model.load_state_dict(e_state)

    corner_model.eval()
    enhancement_model.eval()

    img = cv2.imread(args.img)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.img}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_size = 512
    canvas, scale, x_off, y_off = preprocess_image(img, target_size=input_size)
    input_tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float()
    input_tensor = input_tensor.unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        corner_output = corner_model(input_tensor)
        norm_corners = extract_coords(corner_output)[0].cpu().numpy()

    corners_canvas = norm_corners * (input_size - 1)
    corners_original = (
        corners_canvas - np.array([x_off, y_off], dtype=np.float32)
    ) / scale
    corners_original = corners_original.astype(np.float32)
    corners_original, is_valid = validate_corners(corners_original, img.shape)

    if not is_valid:
        print("WARNING: Invalid corner prediction")

    tl, tr, br, bl = corners_original
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)
    target_width = max(int(top_width), int(bottom_width), 1)
    target_height = max(int(left_height), int(right_height), 1)

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(corners_original, destination)
    rectified = cv2.warpPerspective(img, H, (target_width, target_height))

    # ----------------------------------------------------------------------
    # FIX: Revert to Sliding Window, strictly matching the training crop size
    # ----------------------------------------------------------------------
    patch_size = 256

    pad_h = math.ceil(target_height / patch_size) * patch_size
    pad_w = math.ceil(target_width / patch_size) * patch_size

    padded_rect = np.ones((pad_h, pad_w, 3), dtype=np.uint8) * 255
    padded_rect[:target_height, :target_width] = rectified

    enhanced_canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)

    for y in range(0, pad_h, patch_size):
        for x in range(0, pad_w, patch_size):
            patch = padded_rect[y : y + patch_size, x : x + patch_size]

            # The division by 255.0 perfectly matches dataset_enhancement.py
            patch_tensor = (
                torch.from_numpy(patch.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .to(device)
                / 255.0
            )

            with torch.no_grad():
                out_tensor = enhancement_model(patch_tensor)

            out_patch = (
                (out_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            enhanced_canvas[y : y + patch_size, x : x + patch_size] = out_patch

    final = enhanced_canvas[:target_height, :target_width]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img)
    ax[0].scatter(corners_original[:, 0], corners_original[:, 1], c="red", s=50)
    ax[0].set_title("Detected Corners")
    ax[1].imshow(rectified)
    ax[1].set_title("Rectified")
    ax[2].imshow(final)
    ax[2].set_title("Enhanced")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("output_result.png", bbox_inches="tight")
print("Saved prediction plot to output_result.png")
