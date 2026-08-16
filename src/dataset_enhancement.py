import os
import cv2
import torch
from torch.utils.data import Dataset


class PrecomputedDocumentDataset(Dataset):
    def __init__(self, data_dir, target_size=(256, 256)):
        self.inputs_dir = os.path.join(data_dir, "inputs")
        self.targets_dir = os.path.join(data_dir, "targets")
        self.image_filenames = sorted(os.listdir(self.inputs_dir))
        self.target_size = target_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        input_path = os.path.join(self.inputs_dir, filename)
        target_path = os.path.join(self.targets_dir, filename)

        degraded_img = cv2.imread(input_path)
        degraded_img = cv2.cvtColor(
            degraded_img,
            cv2.COLOR_BGR2RGB,
        ) / 255.0

        clean_target = cv2.imread(target_path)
        clean_target = cv2.cvtColor(
            clean_target,
            cv2.COLOR_BGR2RGB,
        ) / 255.0

        # Image tensors: HWC -> CHW, values normalized to [0, 1]
        degraded_tensor = torch.from_numpy(
            degraded_img.transpose(2, 0, 1)
        ).float()

        clean_target_tensor = torch.from_numpy(
            clean_target.transpose(2, 0, 1)
        ).float()

        return {
            "clean_target": clean_target_tensor,
            "degraded_rectified": degraded_tensor,
        }