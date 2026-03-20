import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from utils import infer_patchwise_tukey_batched


# ------------------------------------------
# Ours Model Builder
# ------------------------------------------
def build_ours(ours_weights, device):
    from model import nonexpansivenn_with_conv
    model = nonexpansivenn_with_conv(
        prox_in=[0.0001,0.001,0.001,0.01,0.1,0.1],
        wavefamilyset=["db4","haar","sym4"],
        input_shape=(64,64),
        grad_in=0.1, levels=3, num_of_layers=10
    ).to(device)

    ckpt = torch.load(ours_weights, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model

# ------------------------------------------
# Utils
# ------------------------------------------
def to_tensor(img_np, device):
    return torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)

def to_numpy_uint8(x):
    return (x.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

# ------------------------------------------
# RUN ON SINGLE IMAGE
# ------------------------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = os.path.dirname(os.path.abspath(__file__))

    # Clean, anonymous, correct paths
    img_path = os.path.join(ROOT, "images", "0030.png")
    ours_weights = os.path.join(ROOT, "models", "sigma15_RGB.pth")
    output_path = os.path.join(ROOT, "outputs", "denoised_output.png")
    sigma = 15

    model = build_ours(ours_weights, device)

    # Load GT image
    img = Image.open(img_path).convert("RGB")
    gt_np = np.float32(img) / 255.

    # Add synthetic Gaussian noise
    noisy_np = np.clip(
        gt_np + np.random.normal(0, sigma/255., gt_np.shape),
        0, 1
    ).astype(np.float32)

    gt = to_tensor(gt_np, device)
    noisy = to_tensor(noisy_np, device)

    # Run our model
    den = infer_patchwise_tukey_batched(model, noisy)

    # Convert to uint8
    den_np = to_numpy_uint8(den)
    gt_np_uint8 = to_numpy_uint8(gt)

    # Compute metrics
    P = psnr_metric(gt_np_uint8, den_np, data_range=255)
    S = ssim_metric(gt_np_uint8, den_np, channel_axis=2, data_range=255)

    print(f"PSNR: {P:.4f} dB")
    print(f"SSIM: {S:.6f}")

    # Save output
    save_image(den, output_path)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()