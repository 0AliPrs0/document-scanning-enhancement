import os
import glob
import math
import random
import cv2
import json
import numpy as np
from tqdm import tqdm


def get_random_perspective_corners(bg_w, bg_h):
    scale = random.uniform(0.30, 0.90)

    doc_w = bg_w * scale
    doc_h = bg_h * scale

    min_cx = int(doc_w / 2)
    max_cx = int(bg_w - doc_w / 2)

    min_cy = int(doc_h / 2)
    max_cy = int(bg_h - doc_h / 2)

    if max_cx <= min_cx:
        cx = bg_w / 2
    else:
        cx = random.randint(min_cx, max_cx)

    if max_cy <= min_cy:
        cy = bg_h / 2
    else:
        cy = random.randint(min_cy, max_cy)

    half_w = doc_w / 2
    half_h = doc_h / 2

    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )

    angle = random.uniform(-75.0, 75.0)

    theta = np.radians(angle)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    rotation_matrix = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ],
        dtype=np.float32,
    )

    rotated_corners = np.dot(
        corners,
        rotation_matrix.T,
    )

    shifted_corners = rotated_corners + np.array([cx, cy], dtype=np.float32)

    wiggle = int(min(bg_w, bg_h) * 0.05)

    for i in range(4):
        shifted_corners[i, 0] += random.randint(
            -wiggle,
            wiggle,
        )
        shifted_corners[i, 1] += random.randint(
            -wiggle,
            wiggle,
        )

    shifted_corners[:, 0] = np.clip(
        shifted_corners[:, 0],
        0,
        bg_w - 1,
    )

    shifted_corners[:, 1] = np.clip(
        shifted_corners[:, 1],
        0,
        bg_h - 1,
    )

    return np.float32(shifted_corners)


def warp_scan_to_background(clean_scan, background_img):
    scan_h, scan_w = clean_scan.shape[:2]
    bg_h, bg_w = background_img.shape[:2]

    src_points = np.float32(
        [
            [0, 0],
            [scan_w - 1, 0],
            [scan_w - 1, scan_h - 1],
            [0, scan_h - 1],
        ]
    )

    dst_points = get_random_perspective_corners(
        bg_w,
        bg_h,
    )

    matrix = cv2.getPerspectiveTransform(
        src_points,
        dst_points,
    )

    warped_scan = cv2.warpPerspective(
        clean_scan,
        matrix,
        (bg_w, bg_h),
    )

    mask = (
        np.ones(
            (scan_h, scan_w),
            dtype=np.uint8,
        )
        * 255
    )

    warped_mask = cv2.warpPerspective(
        mask,
        matrix,
        (bg_w, bg_h),
    )

    warped_mask = warped_mask.astype(np.float32) / 255.0

    warped_mask = cv2.GaussianBlur(
        warped_mask,
        (5, 5),
        0,
    )

    warped_mask = warped_mask[:, :, None]

    background_float = background_img.astype(np.float32)

    warped_float = warped_scan.astype(np.float32)

    result = warped_float * warped_mask + background_float * (1.0 - warped_mask)

    result = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    return result, dst_points


def apply_resolution_loss(img):
    h, w = img.shape[:2]

    scale_factor = random.uniform(
        1.2,
        2.5,
    )

    small_w = max(
        1,
        int(w / scale_factor),
    )

    small_h = max(
        1,
        int(h / scale_factor),
    )

    small_img = cv2.resize(
        img,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.resize(
        small_img,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_color_and_lighting(img_float):
    alpha = random.uniform(
        0.85,
        1.15,
    )

    beta = random.randint(
        -15,
        15,
    )

    img_float = cv2.convertScaleAbs(
        img_float,
        alpha=alpha,
        beta=beta,
    ).astype(np.float32)

    r_scale = random.uniform(
        0.90,
        1.10,
    )

    b_scale = random.uniform(
        0.90,
        1.10,
    )

    img_float[:, :, 2] *= r_scale
    img_float[:, :, 0] *= b_scale

    return np.clip(
        img_float,
        0,
        255,
    )


def apply_shadows_and_gradients(
    img_float,
    corners,
):
    h, w = img_float.shape[:2]

    shadow_mask = np.ones(
        (h, w),
        dtype=np.float32,
    )

    def add_corner_shadow(mask, corner_index):
        corner = corners[corner_index].astype(np.float32)

        prev_corner = corners[(corner_index - 1) % 4].astype(np.float32)

        next_corner = corners[(corner_index + 1) % 4].astype(np.float32)

        v1 = prev_corner - corner
        v2 = next_corner - corner

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1.0 or len2 < 1.0:
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
            edge_dir = np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        else:
            edge_dir /= edge_norm

        depth = random.uniform(0.10, 0.32) * min(len1, len2)

        width = random.uniform(0.25, 0.65) * min(len1, len2)

        center_offset = random.uniform(0.10, 0.35) * depth

        center = corner + inward * center_offset

        center[0] = np.clip(
            center[0],
            0,
            w - 1,
        )

        center[1] = np.clip(
            center[1],
            0,
            h - 1,
        )

        angle = np.degrees(
            np.arctan2(
                edge_dir[1],
                edge_dir[0],
            )
        )

        local_mask = np.zeros(
            (h, w),
            dtype=np.float32,
        )

        ellipse_axes = (
            max(10, int(width)),
            max(10, int(depth)),
        )

        cv2.ellipse(
            local_mask,
            (
                int(center[0]),
                int(center[1]),
            ),
            ellipse_axes,
            angle,
            0,
            360,
            1.0,
            -1,
        )

        blur_size = random.choice([31, 51, 71, 101])

        local_mask = cv2.GaussianBlur(
            local_mask,
            (blur_size, blur_size),
            0,
        )

        intensity = random.uniform(
            0.45,
            0.82,
        )

        mask *= 1.0 - local_mask * intensity

    corner_shadow_probability = random.random()

    if corner_shadow_probability < 0.70:
        if random.random() < 0.70:
            num_corner_shadows = 1
        else:
            num_corner_shadows = 2

        selected_corners = random.sample(
            range(4),
            num_corner_shadows,
        )

        for corner_index in selected_corners:
            add_corner_shadow(
                shadow_mask,
                corner_index,
            )

    if random.random() < 0.45:
        num_points = random.randint(
            3,
            6,
        )

        points = [
            [
                random.randint(0, w - 1),
                random.randint(0, h - 1),
            ]
            for _ in range(num_points)
        ]

        pts = np.array(
            points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        general_shadow = np.ones(
            (h, w),
            dtype=np.float32,
        )

        shadow_intensity = random.uniform(
            0.70,
            0.92,
        )

        cv2.fillPoly(
            general_shadow,
            [pts],
            shadow_intensity,
        )

        general_shadow = cv2.GaussianBlur(
            general_shadow,
            (101, 101),
            50,
        )

        shadow_mask *= general_shadow

    if random.random() < 0.50:
        start = random.uniform(
            0.90,
            1.00,
        )

        end = random.uniform(
            0.90,
            1.00,
        )

        if random.random() < 0.5:
            gradient = np.linspace(
                start,
                end,
                w,
                dtype=np.float32,
            )

            gradient = np.tile(
                gradient,
                (h, 1),
            )
        else:
            gradient = np.linspace(
                start,
                end,
                h,
                dtype=np.float32,
            )

            gradient = np.tile(
                gradient[:, None],
                (1, w),
            )

        shadow_mask *= gradient

    shadow_mask = shadow_mask[:, :, None]

    img_float *= shadow_mask

    return np.clip(
        img_float,
        0,
        255,
    )


def apply_blur_and_noise(img_float):
    blur_probability = random.random()

    if blur_probability < 0.75:
        img_float = cv2.GaussianBlur(
            img_float,
            (3, 3),
            0,
        )
    else:
        kernel_size = 5

        motion_kernel = np.zeros(
            (kernel_size, kernel_size),
            dtype=np.float32,
        )

        motion_kernel[
            kernel_size // 2,
            :,
        ] = (
            1.0 / kernel_size
        )

        img_float = cv2.filter2D(
            img_float,
            -1,
            motion_kernel,
        )

    noise_std = random.uniform(
        1.0,
        5.0,
    )

    noise = np.random.normal(
        0,
        noise_std,
        img_float.shape,
    )

    img_float += noise

    return np.clip(
        img_float,
        0,
        255,
    ).astype(np.uint8)


def apply_jpeg_compression(img_uint8):
    quality = random.randint(
        50,
        95,
    )

    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality,
    ]

    success, encoded_img = cv2.imencode(
        ".jpg",
        img_uint8,
        encode_param,
    )

    if not success:
        return img_uint8

    decoded = cv2.imdecode(
        encoded_img,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        return img_uint8

    return decoded


def order_points(pts):
    center = np.mean(
        pts,
        axis=0,
    )

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0],
    )

    sorted_indices = np.argsort(angles)
    ordered = pts[sorted_indices]

    top_left_index = np.argmin(ordered[:, 0] + ordered[:, 1])

    ordered = np.roll(
        ordered,
        -top_left_index,
        axis=0,
    )

    return ordered.astype(np.float32)


def generate_synthetic_sample(
    clean_scan,
    background_img,
):
    warped_composite, corners = warp_scan_to_background(
        clean_scan,
        background_img,
    )

    img = apply_resolution_loss(warped_composite)

    img_float = img.astype(np.float32)

    img_float = apply_color_and_lighting(img_float)

    img_float = apply_shadows_and_gradients(
        img_float,
        corners,
    )

    img_uint8 = apply_blur_and_noise(img_float)

    final_img = apply_jpeg_compression(img_uint8)

    ordered_corners = order_points(corners)

    return final_img, ordered_corners


def resize_with_padding_and_corners(
    img,
    corners,
    target_size=(512, 512),
):
    target_w, target_h = target_size

    h, w = img.shape[:2]

    scale = min(
        target_w / w,
        target_h / h,
    )

    new_w = max(
        1,
        int(w * scale),
    )

    new_h = max(
        1,
        int(h * scale),
    )

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = (
        np.ones(
            (target_h, target_w, 3),
            dtype=np.uint8,
        )
        * 255
    )

    x_off = (target_w - new_w) // 2

    y_off = (target_h - new_h) // 2

    canvas[
        y_off : y_off + new_h,
        x_off : x_off + new_w,
    ] = resized

    new_corners = corners.copy()

    new_corners[:, 0] = new_corners[:, 0] * scale + x_off

    new_corners[:, 1] = new_corners[:, 1] * scale + y_off

    new_corners[:, 0] /= target_w - 1

    new_corners[:, 1] /= target_h - 1

    new_corners = np.clip(
        new_corners,
        0.0,
        1.0,
    )

    return canvas, new_corners


def precompute_corner_dataset(
    train_data_dir,
    bg_data_dir,
    output_dir,
    total_target_samples=1000,
    target_size=(512, 512),
):
    input_dir = os.path.join(
        output_dir,
        "inputs",
    )

    os.makedirs(
        input_dir,
        exist_ok=True,
    )

    clean_scans = sorted(
        glob.glob(
            os.path.join(
                train_data_dir,
                "*.jpg",
            )
        )
    )

    backgrounds = sorted(
        glob.glob(
            os.path.join(
                bg_data_dir,
                "*.jpg",
            )
        )
    )

    if len(clean_scans) == 0:
        raise RuntimeError("No clean scan images found.")

    if len(backgrounds) == 0:
        raise RuntimeError("No background images found.")

    samples_per_image = math.ceil(total_target_samples / len(clean_scans))

    sample_idx = 0
    labels_dict = {}

    for scan_path in tqdm(clean_scans):
        clean_scan = cv2.imread(
            scan_path,
            cv2.IMREAD_COLOR,
        )

        if clean_scan is None:
            continue

        clean_scan = cv2.cvtColor(
            clean_scan,
            cv2.COLOR_BGR2RGB,
        )

        for _ in range(samples_per_image):
            if sample_idx >= total_target_samples:
                break

            bg_path = random.choice(backgrounds)

            bg = cv2.imread(
                bg_path,
                cv2.IMREAD_COLOR,
            )

            if bg is None:
                continue

            bg = cv2.cvtColor(
                bg,
                cv2.COLOR_BGR2RGB,
            )

            bg_h, bg_w = bg.shape[:2]

            bg_scale = max(
                512 / bg_w,
                512 / bg_h,
            )

            if bg_scale > 1:
                new_w = int(bg_w * bg_scale)

                new_h = int(bg_h * bg_scale)

                bg = cv2.resize(
                    bg,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            raw_photo, corners = generate_synthetic_sample(
                clean_scan,
                bg,
            )

            final_input, final_corners = resize_with_padding_and_corners(
                raw_photo,
                corners,
                target_size,
            )

            file_name = f"sample_{sample_idx:05d}.jpg"

            input_path = os.path.join(
                input_dir,
                file_name,
            )

            cv2.imwrite(
                input_path,
                cv2.cvtColor(
                    final_input,
                    cv2.COLOR_RGB2BGR,
                ),
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    90,
                ],
            )

            labels_dict[file_name] = final_corners.tolist()

            sample_idx += 1

    with open(
        os.path.join(
            output_dir,
            "labels.json",
        ),
        "w",
    ) as f:
        json.dump(
            labels_dict,
            f,
            indent=4,
        )

    print(f"Created {sample_idx} samples.")


if __name__ == "__main__":
    splits = {
        "train": {
            "dir": "data/dataset/train",
            "samples": 800,
        },
        "valid": {
            "dir": "data/dataset/valid",
            "samples": 100,
        },
        "test": {
            "dir": "data/dataset/test",
            "samples": 100,
        },
    }

    for split_name, config in splits.items():
        print(f"Precomputing corners for {split_name}...")

        precompute_corner_dataset(
            train_data_dir=config["dir"],
            bg_data_dir="data/background",
            output_dir=(f"data/precomputed_corners/" f"{split_name}"),
            total_target_samples=config["samples"],
            target_size=(512, 512),
        )
