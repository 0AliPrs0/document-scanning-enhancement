import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BaseCornerDataset(Dataset):
    def __init__(self, data_dir):
        self.inputs_dir = os.path.join(data_dir, "inputs")
        self.labels_path = os.path.join(data_dir, "labels.json")

        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)

        self.image_filenames = sorted(list(self.labels.keys()))

    def __len__(self):
        return len(self.image_filenames)

    def get_raw_data(self, idx):
        filename = self.image_filenames[idx]
        input_path = os.path.join(self.inputs_dir, filename)

        img = cv2.imread(input_path)

        if img is None:
            raise RuntimeError(f"Could not read image: {input_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        corners = np.array(self.labels[filename], dtype=np.float32)

        return img, corners


class RegressionCornerDataset(BaseCornerDataset):
    def __getitem__(self, idx):
        img, corners = self.get_raw_data(idx)

        # Image: HWC -> CHW, corners: (4, 2) -> (8,)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        corners_flat = torch.from_numpy(corners.flatten()).float()

        return {
            "image": img_tensor,
            "corners": corners_flat,
        }


class HeatmapCornerDataset(Dataset):
    def __init__(self, dataset_dir, heatmap_size=256, sigma=7.0):
        self.dataset_dir = dataset_dir
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.inputs_dir = os.path.join(dataset_dir, "inputs")
        self.labels_path = os.path.join(dataset_dir, "labels.json")

        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)

        self.image_filenames = sorted(list(self.labels.keys()))

        if len(self.image_filenames) == 0:
            raise RuntimeError(f"No images found in {self.inputs_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def generate_heatmaps(self, corners):
        # Output: 4 heatmaps with shape (4, heatmap_size, heatmap_size)
        heatmaps = np.zeros(
            (4, self.heatmap_size, self.heatmap_size),
            dtype=np.float32,
        )

        yy, xx = np.meshgrid(
            np.arange(self.heatmap_size),
            np.arange(self.heatmap_size),
            indexing="ij",
        )

        for i in range(4):
            x = float(corners[i, 0]) * (self.heatmap_size - 1)
            y = float(corners[i, 1]) * (self.heatmap_size - 1)

            distance_sq = (xx - x) ** 2 + (yy - y) ** 2
            heatmaps[i] = np.exp(-distance_sq / (2.0 * self.sigma * self.sigma))

        return heatmaps

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.inputs_dir, filename)

        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corners = np.array(self.labels[filename], dtype=np.float32)
        heatmaps = self.generate_heatmaps(corners)

        # Image: HWC -> CHW, values normalized to [0, 1]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        corners_tensor = torch.from_numpy(corners).float()
        heatmaps_tensor = torch.from_numpy(heatmaps).float()

        return {
            "image": image_tensor,
            "corners": corners_tensor,
            "heatmaps": heatmaps_tensor,
        }
