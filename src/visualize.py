import os
import json
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def visualize_corners(
    base_dir="data/precomputed_corners/train",
    num_samples=3,
):
    inputs_dir = os.path.join(base_dir, "inputs")
    labels_path = os.path.join(base_dir, "labels.json")

    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        return

    with open(labels_path, "r") as f:
        labels = json.load(f)

    filenames = list(labels.keys())
    random.shuffle(filenames)
    filenames = filenames[:num_samples]

    for filename in filenames:
        img_path = os.path.join(inputs_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        corners = np.array(labels[filename], dtype=np.float32)
        corners[:, 0] *= w - 1
        corners[:, 1] *= h - 1

        plt.figure(figsize=(8, 8))
        plt.imshow(img)

        plt.scatter(
            corners[:, 0],
            corners[:, 1],
            c="red",
            s=100,
            marker="X",
            edgecolors="white",
            linewidths=1.5,
        )

        polygon = np.vstack((corners, corners[0]))
        plt.plot(
            polygon[:, 0],
            polygon[:, 1],
            c="lime",
            linewidth=2,
        )

        for i, (x, y) in enumerate(corners):
            plt.text(
                x + 10,
                y + 10,
                str(i),
                color="yellow",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    facecolor="red",
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            )

        plt.title(f"Corners: {filename}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def visualize_enhancement(
    base_dir="data/precomputed_enhancements/train",
    num_samples=3,
):
    inputs_dir = os.path.join(base_dir, "inputs")
    targets_dir = os.path.join(base_dir, "targets")

    if not os.path.exists(inputs_dir) or not os.path.exists(targets_dir):
        print(f"Enhancement dataset not found: {base_dir}")
        return

    filenames = os.listdir(inputs_dir)
    random.shuffle(filenames)
    filenames = filenames[:num_samples]

    for filename in filenames:
        input_path = os.path.join(inputs_dir, filename)
        target_path = os.path.join(targets_dir, filename)

        inp_img = cv2.imread(input_path)
        tgt_img = cv2.imread(target_path)

        if inp_img is None or tgt_img is None:
            continue

        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(inp_img)
        axes[0].set_title(f"Degraded Input\n{filename}")
        axes[0].axis("off")

        axes[1].imshow(tgt_img)
        axes[1].set_title(f"Clean Target\n{filename}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualize_corners(
        base_dir="data/precomputed_corners/train",
        num_samples=3,
    )

    visualize_enhancement(
        base_dir="data/precomputed_enhancements/train",
        num_samples=3,
    )
