import os
import glob
import math
import random
import cv2
import json
import numpy as np
from tqdm import tqdm
import argparse


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def get_random_perspective_corners(bg_w, bg_h):
    scale = random.uniform(0.40, 0.85)
    doc_w = bg_w * scale
    doc_h = bg_h * scale
    min_cx = int(doc_w / 2)
    max_cx = int(bg_w - doc_w / 2)
    min_cy = int(doc_h / 2)
    max_cy = int(bg_h - doc_h / 2)
    cx = bg_w / 2 if max_cx <= min_cx else random.randint(min_cx, max_cx)
    cy = bg_h / 2 if max_cy <= min_cy else random.randint(min_cy, max_cy)
    half_w = doc_w / 2
    half_h = doc_h / 2
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
        dtype=np.float32,
    )
    angle = random.uniform(-45.0, 45.0)
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    rotated = np.dot(corners, rotation_matrix.T)
    shifted = rotated + np.array([cx, cy], dtype=np.float32)
    wiggle = int(min(bg_w, bg_h) * 0.08)
    for i in range(4):
        shifted[i, 0] += random.uniform(-wiggle, wiggle)
        shifted[i, 1] += random.uniform(-wiggle, wiggle)
    shifted[:, 0] = np.clip(shifted[:, 0], 2, bg_w - 3)
    shifted[:, 1] = np.clip(shifted[:, 1], 2, bg_h - 3)
    shifted = order_points(shifted)
    return shifted


def warp_scan_to_background(clean_scan, background_img):
    scan_h, scan_w = clean_scan.shape[:2]
    bg_h, bg_w = background_img.shape[:2]
    src_points = np.float32(
        [[0, 0], [scan_w - 1, 0], [scan_w - 1, scan_h - 1], [0, scan_h - 1]]
    )
    dst_points = get_random_perspective_corners(bg_w, bg_h)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_scan = cv2.warpPerspective(clean_scan, matrix, (bg_w, bg_h))
    mask = np.ones((scan_h, scan_w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, matrix, (bg_w, bg_h))
    warped_mask = warped_mask.astype(np.float32) / 255.0
    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 0)
    warped_mask = warped_mask[:, :, None]
    background_float = background_img.astype(np.float32)
    warped_float = warped_scan.astype(np.float32)
    result = warped_float * warped_mask + background_float * (1.0 - warped_mask)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, dst_points


def apply_resolution_loss(img):
    h, w = img.shape[:2]
    scale_factor = random.uniform(1.2, 2.2)
    small_w = max(1, int(w / scale_factor))
    small_h = max(1, int(h / scale_factor))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_color_and_lighting(img):
    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-12, 12)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta).astype(np.float32)
    r_scale = random.uniform(0.95, 1.05)
    b_scale = random.uniform(0.95, 1.05)
    out[:, :, 2] *= r_scale
    out[:, :, 0] *= b_scale
    return np.clip(out, 0, 255)


def apply_shadows_and_gradients(img, corners):
    h, w = img.shape[:2]
    shadow_mask = np.ones((h, w), dtype=np.float32)

    def add_corner_shadow(mask, corner_index):
        corner = corners[corner_index].astype(np.float32)
        prev_corner = corners[(corner_index - 1) % 4].astype(np.float32)
        next_corner = corners[(corner_index + 1) % 4].astype(np.float32)
        v1 = prev_corner - corner
        v2 = next_corner - corner
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 5 or len2 < 5:
            return
        v1 /= len1
        v2 /= len2
        inward = v1 + v2
        inward_norm = np.linalg.norm(inward)
        if inward_norm < 1e-6:
            return
        inward /= inward_norm
        edge_dir = v1 - v2
        edge_norm = np.linalg.norm(edge_dir)
        if edge_norm < 1e-6:
            return
        edge_dir /= edge_norm
        base_size = min(len1, len2)
        depth = random.uniform(0.08, 0.22) * base_size
        width = random.uniform(0.18, 0.45) * base_size
        center_offset = random.uniform(0.05, 0.20) * depth
        center = corner + inward * center_offset
        center[0] = np.clip(center[0], 0, w - 1)
        center[1] = np.clip(center[1], 0, h - 1)
        angle = np.degrees(np.arctan2(edge_dir[1], edge_dir[0]))
        local_mask = np.zeros((h, w), dtype=np.float32)
        axes = (max(8, int(width)), max(8, int(depth)))
        cv2.ellipse(
            local_mask, (int(center[0]), int(center[1])), axes, angle, 0, 360, 1.0, -1
        )
        blur_size = random.choice([31, 51, 71])
        local_mask = cv2.GaussianBlur(local_mask, (blur_size, blur_size), 0)
        intensity = random.uniform(0.30, 0.60)
        mask *= 1.0 - local_mask * intensity

    if random.random() < 0.75:
        count = 1 if random.random() < 0.65 else 2
        selected = random.sample(range(4), count)
        for corner_index in selected:
            add_corner_shadow(shadow_mask, corner_index)
    if random.random() < 0.30:
        gradient_start = random.uniform(0.90, 1.00)
        gradient_end = random.uniform(0.90, 1.00)
        if random.random() < 0.5:
            gradient = np.linspace(gradient_start, gradient_end, w, dtype=np.float32)
            gradient = np.tile(gradient, (h, 1))
        else:
            gradient = np.linspace(gradient_start, gradient_end, h, dtype=np.float32)
            gradient = np.tile(gradient[:, None], (1, w))
        shadow_mask *= gradient
    shadow_mask = shadow_mask[:, :, None]
    img = img * shadow_mask
    return np.clip(img, 0, 255)


def apply_blur_and_noise(img):
    if random.random() < 0.70:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = 1.0 / 5.0
        img = cv2.filter2D(img, -1, kernel)
    noise_std = random.uniform(0.5, 3.0)
    noise = np.random.normal(0, noise_std, img.shape)
    img = img + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img):
    quality = random.randint(60, 95)
    success, encoded = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return img
    return decoded


def generate_synthetic_sample(clean_scan, background_img):
    warped, corners = warp_scan_to_background(clean_scan, background_img)
    degraded = apply_resolution_loss(warped)
    degraded = degraded.astype(np.float32)
    degraded = apply_color_and_lighting(degraded)
    degraded = apply_shadows_and_gradients(degraded, corners)
    degraded = apply_blur_and_noise(degraded)
    degraded = apply_jpeg_compression(degraded)
    return degraded, corners


def resize_with_padding(img, target_size=(512, 512)):
    target_w, target_h = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def precompute_e2e_dataset(
    clean_data_dir, background_dir, output_dir, total_samples, target_size=(512, 512)
):
    raw_dir = os.path.join(output_dir, "raw_inputs")
    clean_dir = os.path.join(output_dir, "clean_targets")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    clean_paths = sorted(glob.glob(os.path.join(clean_data_dir, "*.jpg")))
    background_paths = sorted(glob.glob(os.path.join(background_dir, "*.jpg")))
    if not clean_paths:
        raise RuntimeError(f"No clean images in {clean_data_dir}")
    if not background_paths:
        raise RuntimeError(f"No background images in {background_dir}")
    labels = {}
    sample_idx = 0
    while sample_idx < total_samples:
        clean_path = random.choice(clean_paths)
        background_path = random.choice(background_paths)
        clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        background = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if clean is None or background is None:
            continue
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background_h, background_w = background.shape[:2]
        if background_w < 512 or background_h < 512:
            scale = max(512 / background_w, 512 / background_h)
            background = cv2.resize(
                background,
                (int(background_w * scale), int(background_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        raw, corners = generate_synthetic_sample(clean, background)
        raw = resize_with_padding(raw, target_size)
        clean_target = resize_with_padding(clean, target_size)
        filename = f"sample_{sample_idx:05d}.jpg"
        cv2.imwrite(
            os.path.join(raw_dir, filename),
            cv2.cvtColor(raw, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        cv2.imwrite(
            os.path.join(clean_dir, filename),
            cv2.cvtColor(clean_target, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
        corners_normalized = corners.copy()
        corners_normalized[:, 0] /= background.shape[1] - 1
        corners_normalized[:, 1] /= background.shape[0] - 1
        corners_normalized[:, 0] = np.clip(corners_normalized[:, 0], 0, 1)
        corners_normalized[:, 1] = np.clip(corners_normalized[:, 1], 0, 1)
        scale_x = target_size[0] / background.shape[1]
        scale_y = target_size[1] / background.shape[0]
        corners_512 = corners.copy()
        corners_512[:, 0] *= scale_x
        corners_512[:, 1] *= scale_y
        corners_512[:, 0] = np.clip(corners_512[:, 0], 0, target_size[0] - 1)
        corners_512[:, 1] = np.clip(corners_512[:, 1], 0, target_size[1] - 1)
        labels[filename] = (
            corners_512
            / (np.array([target_size[0] - 1, target_size[1] - 1], dtype=np.float32))
        ).tolist()
        sample_idx += 1
    with open(os.path.join(output_dir, "corners.json"), "w") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--bg_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    precompute_e2e_dataset(args.clean_dir, args.bg_dir, args.out_dir, args.samples)
