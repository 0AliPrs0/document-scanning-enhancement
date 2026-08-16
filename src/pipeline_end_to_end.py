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
from model_corners import CornerHeatmapUNet, CornerDirectRegressor


def extract_coords(logits, temperature=20.0):
    B, C, H, W = logits.shape

    flat = logits.view(B, C, -1)
    max_idx = torch.argmax(flat, dim=2)

    y_coords = (max_idx // W).float() / (H - 1)
    x_coords = (max_idx % W).float() / (W - 1)

    return torch.stack(
        [x_coords, y_coords],
        dim=2,
    )


def load_weights(model, path, device):
    state = torch.load(
        path,
        map_location=device,
    )

    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img", type=str, required=True, help="Path to the input image")

    args = parser.parse_args()

    corner_weights_path = "output/best_heatmap_dropout.pth"
    enh_weights_path = "output/best_enhancement_dropout.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corner_name = os.path.basename(corner_weights_path).lower()

    corner_is_reg = "reg" in corner_name

    corner_dropout = "drop" in corner_name and "nodrop" not in corner_name

    if corner_is_reg:
        corner_model = CornerDirectRegressor(corner_dropout).to(device)
    else:
        corner_model = CornerHeatmapUNet(corner_dropout).to(device)

    corner_model = load_weights(
        corner_model,
        corner_weights_path,
        device,
    )
    corner_model.eval()

    enh_name = os.path.basename(enh_weights_path).lower()

    enh_dropout = "drop" in enh_name and "nodrop" not in enh_name

    enhancement_model = DocumentEnhancementUNet(enh_dropout).to(device)

    enhancement_model = load_weights(
        enhancement_model,
        enh_weights_path,
        device,
    )
    enhancement_model.eval()

    img = cv2.imread(args.img)

    if img is None:
        raise RuntimeError(f"Could not read image: {args.img}")

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )

    h, w = img.shape[:2]

    input_size = 512

    scale = min(
        input_size / w,
        input_size / h,
    )

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    canvas = (
        np.ones(
            (input_size, input_size, 3),
            dtype=np.uint8,
        )
        * 255
    )

    x_off = (input_size - nw) // 2
    y_off = (input_size - nh) // 2

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    canvas[
        y_off : y_off + nh,
        x_off : x_off + nw,
    ] = resized

    input_tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float()

    input_tensor = input_tensor.unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        corner_output = corner_model(input_tensor)

        if corner_is_reg:
            # Output: (B, 8) -> 4 normalized (x, y) coordinates.
            norm_corners = corner_output[0].cpu().numpy().reshape(4, 2)
        else:
            # Output: (B, 4, H, W) -> 4 corner heatmaps.
            norm_corners = extract_coords(corner_output)[0].cpu().numpy()

    corners_canvas = norm_corners * input_size

    corners_original = (
        corners_canvas
        - np.array(
            [x_off, y_off],
            dtype=np.float32,
        )
    ) / scale

    corners_original = corners_original.astype(np.float32)

    tl, tr, br, bl = corners_original

    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)

    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)

    target_width = max(
        int(top_width),
        int(bottom_width),
        1,
    )

    target_height = max(
        int(left_height),
        int(right_height),
        1,
    )

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(
        corners_original,
        destination,
    )

    rectified = cv2.warpPerspective(
        img,
        H,
        (target_width, target_height),
    )

    patch_size = 512

    pad_h = math.ceil(target_height / patch_size) * patch_size

    pad_w = math.ceil(target_width / patch_size) * patch_size

    padded_rect = (
        np.ones(
            (pad_h, pad_w, 3),
            dtype=np.uint8,
        )
        * 255
    )

    padded_rect[
        :target_height,
        :target_width,
    ] = rectified

    enhanced_canvas = np.zeros(
        (pad_h, pad_w, 3),
        dtype=np.uint8,
    )

    for y in range(0, pad_h, patch_size):
        for x in range(0, pad_w, patch_size):
            patch = padded_rect[
                y : y + patch_size,
                x : x + patch_size,
            ]

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

            enhanced_canvas[
                y : y + patch_size,
                x : x + patch_size,
            ] = out_patch

    final = enhanced_canvas[
        :target_height,
        :target_width,
    ]

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    ax[0].imshow(img)

    ax[0].scatter(
        corners_original[:, 0],
        corners_original[:, 1],
        c="red",
        s=50,
    )

    polygon = np.vstack((corners_original, corners_original[0]))

    ax[0].plot(
        polygon[:, 0],
        polygon[:, 1],
        c="lime",
        linewidth=2,
        linestyle="--",
    )

    ax[0].set_title("Detected Corners")
    ax[1].imshow(rectified)
    ax[1].set_title("Rectified")
    ax[2].imshow(final)
    ax[2].set_title("Enhanced")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()