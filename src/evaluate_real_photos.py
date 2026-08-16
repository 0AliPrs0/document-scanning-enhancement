import os
import sys
import cv2
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_enhancement import DocumentEnhancementUNet


def get_ocr_confidence(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ocr_data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT,
    )

    confidences = [
        int(conf)
        for conf in ocr_data["conf"]
        if str(conf) != "-1" and str(conf).strip() != ""
    ]

    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)


def enhance_image(model, img_rgb, patch_size=512, device="cpu"):
    orig_h, orig_w = img_rgb.shape[:2]

    pad_h = math.ceil(orig_h / patch_size) * patch_size
    pad_w = math.ceil(orig_w / patch_size) * patch_size

    padded_img = np.ones((pad_h, pad_w, 3), dtype=np.uint8) * 255
    padded_img[:orig_h, :orig_w] = img_rgb

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

    return enhanced_canvas[:orig_h, :orig_w]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the rectified input image",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to the commercial scan reference",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="output/best_enhancement_nodrop.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--use_dropout",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = DocumentEnhancementUNet(use_dropout=args.use_dropout).to(device)

    state_dict = torch.load(
        args.weights,
        map_location=device,
    )

    # Remove DataParallel prefix when loading multi-GPU weights.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    inp_img = cv2.imread(args.input)
    ref_img = cv2.imread(args.reference)

    if inp_img is None:
        raise ValueError(f"Could not read input image: {args.input}")

    if ref_img is None:
        raise ValueError(f"Could not read reference image: {args.reference}")

    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = inp_img.shape[:2]

    ref_img = cv2.resize(
        ref_img,
        (orig_w, orig_h),
        interpolation=cv2.INTER_AREA,
    )

    print("[INFO] Enhancing image with model...")
    enhanced_img = enhance_image(
        model,
        inp_img,
        device=device,
    )

    print("[INFO] Running OCR for Readability evaluation...")

    conf_inp = get_ocr_confidence(inp_img)
    conf_enh = get_ocr_confidence(enhanced_img)
    conf_ref = get_ocr_confidence(ref_img)

    print("\n" + "=" * 40)
    print("OCR Confidence Scores (0-100%):")
    print("-" * 40)
    # print(f"Rectified Input:  {conf_inp:.2f}%")
    print(f"Model Output:     {conf_enh:.2f}%")
    print(f"CamScanner Ref:   {conf_ref:.2f}%")
    print("=" * 40 + "\n")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 8),
    )

    axes[0].imshow(inp_img)
    # axes[0].set_title(
    #     f"1. Rectified Input\nOCR Conf: {conf_inp:.1f}%",
    #     fontsize=14,
    # )
    axes[0].axis("off")

    axes[1].imshow(enhanced_img)
    axes[1].set_title(
        f"2. Your Model Output\nOCR Conf: {conf_enh:.1f}%",
        fontsize=14,
        color="blue",
    )
    axes[1].axis("off")

    axes[2].imshow(ref_img)
    axes[2].set_title(
        f"3. CamScanner Reference\nOCR Conf: {conf_ref:.1f}%",
        fontsize=14,
        color="green",
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
