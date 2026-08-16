import math
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


def calculate_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)

    if mse == 0:
        return float("inf")

    return 10 * math.log10((max_val**2) / mse.item())


def evaluate_model(model, dataloader, device):
    model.eval()

    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_base_psnr = 0.0
    total_base_ssim = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["degraded_rectified"].to(device)
            targets = batch["clean_target"].to(device)

            outputs = model(inputs)

            total_psnr += calculate_psnr(outputs, targets)
            total_ssim += ssim_module(outputs, targets).item()

            total_base_psnr += calculate_psnr(inputs, targets)
            total_base_ssim += ssim_module(inputs, targets).item()

    num_batches = len(dataloader)

    return (
        total_psnr / num_batches,
        total_ssim / num_batches,
        total_base_psnr / num_batches,
        total_base_ssim / num_batches,
    )
