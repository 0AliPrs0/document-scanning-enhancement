import os
import glob
import math
import cv2
import numpy as np
from tqdm import tqdm


def resize_with_padding(img, target_size=(512, 512)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

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

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[
        y : y + new_h,
        x : x + new_w,
    ] = resized

    return canvas


def apply_opencv_degradations(img):
    h, w = img.shape[:2]

    scale = np.random.uniform(0.25, 0.5)

    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))

    small = cv2.resize(
        img,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    )

    img = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    alpha = np.random.uniform(0.90, 1.10)
    beta = np.random.uniform(-10, 10)

    img = cv2.convertScaleAbs(
        img,
        alpha=alpha,
        beta=beta,
    )

    img = img.astype(np.float32)

    r_factor = np.random.uniform(0.95, 1.05)
    b_factor = np.random.uniform(0.95, 1.05)

    img[:, :, 0] *= r_factor
    img[:, :, 2] *= b_factor

    img = np.clip(img, 0, 255)

    start = np.random.uniform(0.80, 1.00)
    end = np.random.uniform(0.80, 1.00)

    if np.random.rand() < 0.5:
        gradient = np.linspace(start, end, w)
        gradient = np.tile(gradient, (h, 1))
    else:
        gradient = np.linspace(start, end, h)
        gradient = np.tile(
            gradient[:, None],
            (1, w),
        )

    gradient = gradient[:, :, None]

    img *= gradient
    img = np.clip(img, 0, 255)

    if np.random.rand() < 0.5:
        shadow = np.ones(
            (h, w),
            dtype=np.float32,
        )

        num_points = np.random.randint(3, 6)

        points = np.array(
            [
                [
                    np.random.randint(0, w),
                    np.random.randint(0, h),
                ]
                for _ in range(num_points)
            ],
            dtype=np.int32,
        )

        shadow_value = np.random.uniform(0.70, 0.90)

        cv2.fillPoly(
            shadow,
            [points],
            shadow_value,
        )

        kernel_size = np.random.choice([31, 51, 71])

        shadow = cv2.GaussianBlur(
            shadow,
            (kernel_size, kernel_size),
            0,
        )

        shadow = shadow[:, :, None]

        img *= shadow

    blur_probability = np.random.rand()

    if blur_probability < 0.35:
        kernel_size = np.random.choice([3, 5])

        img = cv2.GaussianBlur(
            img,
            (kernel_size, kernel_size),
            0,
        )

    elif blur_probability < 0.50:
        kernel_size = np.random.choice([3, 5])

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

        img = cv2.filter2D(
            img,
            -1,
            motion_kernel,
        )

    noise_std = np.random.uniform(1.5, 5.0)

    noise = np.random.normal(
        0,
        noise_std,
        img.shape,
    ).astype(np.float32)

    img = img + noise
    img = np.clip(
        img,
        0,
        255,
    ).astype(np.uint8)

    quality = np.random.randint(30, 81)

    success, encoded = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if success:
        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if decoded is not None:
            img = decoded

    return img


def create_perspective_photo(
    clean_scan,
    background,
):
    scan_h, scan_w = clean_scan.shape[:2]
    bg_h, bg_w = background.shape[:2]

    margin_x = int(bg_w * 0.15)
    margin_y = int(bg_h * 0.15)

    left_x = np.random.randint(
        0,
        max(1, margin_x),
    )

    right_x = np.random.randint(
        max(1, bg_w - margin_x),
        bg_w,
    )

    top_y = np.random.randint(
        0,
        max(1, margin_y),
    )

    bottom_y = np.random.randint(
        max(1, bg_h - margin_y),
        bg_h,
    )

    jitter_x = max(1, int(bg_w * 0.05))
    jitter_y = max(1, int(bg_h * 0.05))

    corners = np.array(
        [
            [
                np.clip(
                    left_x
                    + np.random.randint(
                        -jitter_x,
                        jitter_x + 1,
                    ),
                    0,
                    bg_w - 1,
                ),
                np.clip(
                    top_y
                    + np.random.randint(
                        -jitter_y,
                        jitter_y + 1,
                    ),
                    0,
                    bg_h - 1,
                ),
            ],
            [
                np.clip(
                    right_x
                    + np.random.randint(
                        -jitter_x,
                        jitter_x + 1,
                    ),
                    0,
                    bg_w - 1,
                ),
                np.clip(
                    top_y
                    + np.random.randint(
                        -jitter_y,
                        jitter_y + 1,
                    ),
                    0,
                    bg_h - 1,
                ),
            ],
            [
                np.clip(
                    right_x
                    + np.random.randint(
                        -jitter_x,
                        jitter_x + 1,
                    ),
                    0,
                    bg_w - 1,
                ),
                np.clip(
                    bottom_y
                    + np.random.randint(
                        -jitter_y,
                        jitter_y + 1,
                    ),
                    0,
                    bg_h - 1,
                ),
            ],
            [
                np.clip(
                    left_x
                    + np.random.randint(
                        -jitter_x,
                        jitter_x + 1,
                    ),
                    0,
                    bg_w - 1,
                ),
                np.clip(
                    bottom_y
                    + np.random.randint(
                        -jitter_y,
                        jitter_y + 1,
                    ),
                    0,
                    bg_h - 1,
                ),
            ],
        ],
        dtype=np.float32,
    )

    src_pts = np.array(
        [
            [0, 0],
            [scan_w - 1, 0],
            [scan_w - 1, scan_h - 1],
            [0, scan_h - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(
        src_pts,
        corners,
    )

    warped_scan = cv2.warpPerspective(
        clean_scan,
        H,
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
        H,
        (bg_w, bg_h),
    )

    warped_mask = warped_mask.astype(np.float32) / 255.0

    warped_mask = cv2.GaussianBlur(
        warped_mask,
        (3, 3),
        0,
    )

    warped_mask = warped_mask[:, :, None]

    raw_photo = warped_scan.astype(np.float32) * warped_mask + background.astype(
        np.float32
    ) * (1.0 - warped_mask)

    raw_photo = np.clip(
        raw_photo,
        0,
        255,
    ).astype(np.uint8)

    return raw_photo, corners


def precompute_dataset(
    train_data_dir,
    output_dir,
    total_target_samples=1000,
    target_size=(512, 512),
):
    input_dir = os.path.join(
        output_dir,
        "inputs",
    )

    target_dir = os.path.join(
        output_dir,
        "targets",
    )

    os.makedirs(
        input_dir,
        exist_ok=True,
    )

    os.makedirs(
        target_dir,
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

    if len(clean_scans) == 0:
        raise ValueError(f"No JPG images found in {train_data_dir}")

    samples_per_image = math.ceil(total_target_samples / len(clean_scans))

    sample_idx = 0

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

        clean_target = resize_with_padding(
            clean_scan,
            target_size,
        )

        for _ in range(samples_per_image):
            if sample_idx >= total_target_samples:
                break

            bg_path = np.random.choice(clean_scans)

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

            raw_photo, corners = create_perspective_photo(
                clean_scan,
                bg,
            )

            degraded_raw_photo = apply_opencv_degradations(raw_photo)

            scan_h, scan_w = clean_scan.shape[:2]

            src_pts = np.array(
                [
                    [0, 0],
                    [scan_w - 1, 0],
                    [scan_w - 1, scan_h - 1],
                    [0, scan_h - 1],
                ],
                dtype=np.float32,
            )

            H_inv = cv2.getPerspectiveTransform(
                corners,
                src_pts,
            )

            rectified_degraded_photo = cv2.warpPerspective(
                degraded_raw_photo,
                H_inv,
                (scan_w, scan_h),
            )

            degraded_input = resize_with_padding(
                rectified_degraded_photo,
                target_size,
            )

            file_name = f"sample_{sample_idx:05d}.jpg"

            cv2.imwrite(
                os.path.join(
                    input_dir,
                    file_name,
                ),
                cv2.cvtColor(
                    degraded_input,
                    cv2.COLOR_RGB2BGR,
                ),
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80,
                ],
            )

            cv2.imwrite(
                os.path.join(
                    target_dir,
                    file_name,
                ),
                cv2.cvtColor(
                    clean_target,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            sample_idx += 1


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
        print(f"Precomputing enhancement data for {split_name}...")

        precompute_dataset(
            train_data_dir=config["dir"],
            output_dir=(f"data/precomputed_enhancements/" f"{split_name}"),
            total_target_samples=config["samples"],
        )
