import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_corners import CornerDirectRegressor, CornerHeatmapUNet


def extract_coords(heatmaps, temperature=20.0):
    B, C, H, W = heatmaps.shape
    eps = 1e-6

    scores = torch.log(heatmaps.clamp(min=eps))
    scores = scores.view(B, C, -1)

    probs = torch.softmax(scores * temperature, dim=2)
    probs = probs.view(B, C, H, W)

    y_grid = torch.linspace(
        0.0,
        1.0,
        H,
        device=heatmaps.device,
    ).view(1, 1, H, 1)

    x_grid = torch.linspace(
        0.0,
        1.0,
        W,
        device=heatmaps.device,
    ).view(1, 1, 1, W)

    x_coords = torch.sum(probs * x_grid, dim=(2, 3))
    y_coords = torch.sum(probs * y_grid, dim=(2, 3))

    return torch.stack([x_coords, y_coords], dim=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=20.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_name = os.path.basename(args.weights).lower()

    is_regression = "reg" in weights_name
    use_dropout = "drop" in weights_name and "nodrop" not in weights_name

    if is_regression:
        model = CornerDirectRegressor(use_dropout).to(device)
    else:
        model = CornerHeatmapUNet(use_dropout).to(device)

    checkpoint = torch.load(
        args.weights,
        map_location=device,
    )

    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model.eval()

    img = cv2.imread(args.img)

    if img is None:
        raise RuntimeError(f"Could not read: {args.img}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    s = args.img_size

    scale = min(s / w, s / h)

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    canvas = (
        np.ones(
            (s, s, 3),
            dtype=np.uint8,
        )
        * 255
    )

    x_off = (s - nw) // 2
    y_off = (s - nh) // 2

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    canvas[
        y_off : y_off + nh,
        x_off : x_off + nw,
    ] = resized

    tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float()

    tensor = tensor.unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(tensor)

        if is_regression:
            # Output: (B, 8) -> 4 normalized (x, y) corner coordinates.
            norm_corners = output[0].cpu().numpy().reshape(4, 2)
        else:
            # Output: (B, 4, H, W) -> 4 corner heatmaps.
            norm_corners = (
                extract_coords(
                    output,
                    temperature=args.temperature,
                )[0]
                .cpu()
                .numpy()
            )

    corners_canvas = norm_corners * (s - 1)

    corners_original = (
        corners_canvas
        - np.array(
            [x_off, y_off],
            dtype=np.float32,
        )
    ) / scale

    corners_original[:, 0] = np.clip(
        corners_original[:, 0],
        0,
        w - 1,
    )

    corners_original[:, 1] = np.clip(
        corners_original[:, 1],
        0,
        h - 1,
    )

    print("Corners:")

    for i, corner in enumerate(corners_original):
        print(f"{i}: " f"x={corner[0]:.2f}, " f"y={corner[1]:.2f}")

    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    plt.scatter(
        corners_original[:, 0],
        corners_original[:, 1],
        c="red",
        s=70,
    )

    polygon = np.vstack((corners_original, corners_original[0]))

    plt.plot(
        polygon[:, 0],
        polygon[:, 1],
        c="lime",
        linewidth=2,
        linestyle="--",
    )

    for i, corner in enumerate(corners_original):
        plt.text(
            corner[0] + 5,
            corner[1] + 5,
            str(i),
            fontsize=14,
            color="yellow",
            fontweight="bold",
            bbox=dict(
                facecolor="red",
                alpha=0.5,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
        )

    plt.axis("off")
    plt.show()
