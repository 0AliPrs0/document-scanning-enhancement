import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_corners import CornerHeatmapUNet, CornerDirectRegressor


def extract_coords(logits, temperature=20.0):
    B, C, H, W = logits.shape
    flat = logits.view(B, C, -1)
    max_idx = torch.argmax(flat, dim=2)
    y_coords = (max_idx // W).float() / (H - 1)
    x_coords = (max_idx % W).float() / (W - 1)
    return torch.stack([x_coords, y_coords], dim=2)


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def order_points(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    ordered = pts[sorted_indices]
    top_left_index = np.argmin(ordered[:, 0] + ordered[:, 1])
    return np.roll(ordered, -top_left_index, axis=0).astype(np.float32)


def get_all_coco_corners(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    filename_to_corners = {}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        filename = img_id_to_filename.get(img_id)
        if not filename:
            continue

        segmentation = ann.get("segmentation")
        if not segmentation:
            continue

        points = np.array(segmentation[0], dtype=np.float32).reshape(-1, 2)
        if len(points) > 4 and np.allclose(points[0], points[-1]):
            points = points[:-1]

        if len(points) == 4:
            filename_to_corners[filename] = order_points(points)

    return filename_to_corners


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir", type=str, required=True, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--json", type=str, required=True, help="Path to the COCO annotation JSON file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the corner detection model weights.",
    )
    parser.add_argument("--img_size", type=int, default=512)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    weights_name = os.path.basename(args.weights).lower()
    is_regression = "reg" in weights_name
    use_dropout = "drop" in weights_name and "nodrop" not in weights_name

    if is_regression:
        model = CornerDirectRegressor(use_dropout).to(device)
        print("[INFO] Loaded Regression Model.")
    else:
        model = CornerHeatmapUNet(use_dropout).to(device)
        print("[INFO] Loaded Heatmap Model.")

    model = load_weights(model, args.weights, device)
    model.eval()

    print("[INFO] Parsing COCO annotations...")
    gt_corners_dict = get_all_coco_corners(args.json)
    print(f"[INFO] Found {len(gt_corners_dict)} valid image annotations.")

    total_error = 0.0
    valid_images = 0

    for filename, annotated_corners in tqdm(gt_corners_dict.items(), desc="Evaluating Images"):
        img_path = os.path.join(args.img_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        s = args.img_size
        scale = min(s / w, s / h)

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        canvas = np.ones((s, s, 3), dtype=np.uint8) * 255

        x_off = (s - nw) // 2
        y_off = (s - nh) // 2

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas[y_off : y_off + nh, x_off : x_off + nw] = resized

        input_tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float()
        input_tensor = input_tensor.unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            output = model(input_tensor)

            if is_regression:
                norm_corners = output[0].cpu().numpy().reshape(4, 2)
            else:
                norm_corners = extract_coords(output)[0].cpu().numpy()

        corners_canvas = norm_corners * (s - 1)

        predicted_corners = (
            corners_canvas - np.array([x_off, y_off], dtype=np.float32)
        ) / scale

        predicted_corners[:, 0] = np.clip(predicted_corners[:, 0], 0, w - 1)
        predicted_corners[:, 1] = np.clip(predicted_corners[:, 1], 0, h - 1)

        predicted_corners = order_points(predicted_corners.astype(np.float32))

        mean_error = np.mean(np.linalg.norm(predicted_corners - annotated_corners, axis=1))
        
        total_error += mean_error
        valid_images += 1

    if valid_images > 0:
        overall_mean_error = total_error / valid_images
        print(f"\n[RESULT] Evaluated {valid_images} images successfully.")
        print(f"[RESULT] Overall Mean Corner Localization Error: {overall_mean_error:.2f} pixels")
    else:
        print("\n[RESULT] No valid images were processed.")

if __name__ == "__main__":
    main()