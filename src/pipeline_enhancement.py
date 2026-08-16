import cv2
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model_enhancement import DocumentEnhancementUNet


def inference_enhancement_pipeline(
    model,
    rectified_image_path,
    patch_size=512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    img = cv2.imread(rectified_image_path)

    if img is None:
        raise ValueError(f"Could not read image: {rectified_image_path}")

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )

    orig_h, orig_w = img.shape[:2]

    pad_h = math.ceil(orig_h / patch_size) * patch_size

    pad_w = math.ceil(orig_w / patch_size) * patch_size

    padded_img = (
        np.ones(
            (pad_h, pad_w, 3),
            dtype=np.uint8,
        )
        * 255
    )

    padded_img[
        :orig_h,
        :orig_w,
    ] = img

    enhanced_canvas = np.zeros(
        (pad_h, pad_w, 3),
        dtype=np.uint8,
    )

    for y in range(0, pad_h, patch_size):
        for x in range(0, pad_w, patch_size):
            patch = padded_img[
                y : y + patch_size,
                x : x + patch_size,
            ]

            input_tensor = (
                torch.from_numpy(patch.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .to(device)
                / 255.0
            )

            with torch.no_grad():
                output_tensor = model(input_tensor)

            out_patch = (
                (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )

            enhanced_canvas[
                y : y + patch_size,
                x : x + patch_size,
            ] = out_patch

    output_img_restored = enhanced_canvas[
        :orig_h,
        :orig_w,
    ]

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 7),
    )

    ax[0].imshow(img)
    ax[0].axis("off")

    ax[1].imshow(output_img_restored)
    ax[1].axis("off")

    plt.show()

    return output_img_restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="best_enhancement_weights.pth",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DocumentEnhancementUNet().to(device)

    state_dict = torch.load(
        args.weights,
        map_location=device,
    )

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    inference_enhancement_pipeline(
        model,
        args.img,
        patch_size=args.img_size,
    )
