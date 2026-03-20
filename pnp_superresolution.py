# ==========================================================
#  Single-Image Super-Resolution
# ==========================================================

import os
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from utils import infer_patchwise_tukey_batched as denoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


# ==========================================================
# Load OUR model 
# ==========================================================
def load_ours(ckpt_path):
    from model import nonexpansivenn_with_conv   

    prox_in = [0.0001, 0.001, 0.001, 0.01, 0.1, 0.1]
    wavefamilyset = ["db4", "haar", "sym4"]

    model = nonexpansivenn_with_conv(
        prox_in=prox_in,
        wavefamilyset=wavefamilyset,
        input_shape=(64, 64),
        grad_in=0.1,
        levels=3,
        num_of_layers=10
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    return model.eval()


# ==========================================================
# Forward operator A  (blur + downsample)
# Adjoint operator AT (upsample + blur)
# ==========================================================
def funcA(im_input, mask, fx, fy, target_shape=None):
    h_target, w_target = target_shape
    out = np.empty((h_target, w_target, 3), dtype=np.float32)
    for c in range(3):
        blur = cv2.filter2D(im_input[..., c], -1, mask)
        out[..., c] = cv2.resize(blur, (w_target, h_target), interpolation=cv2.INTER_CUBIC)
    return out

def funcAtranspose(im_input, mask, fx, fy, target_shape=None):
    h_target, w_target = target_shape
    out = np.empty((h_target, w_target, 3), dtype=np.float32)
    for c in range(3):
        up = cv2.resize(im_input[..., c], (w_target, h_target), interpolation=cv2.INTER_CUBIC)
        out[..., c] = cv2.filter2D(up, -1, mask)
    return out

def proj(x):
    return np.clip(x, 0, 1)

def crop_to_multiple(img, m=8):
    H, W = img.shape[:2]
    return img[:(H//m)*m, :(W//m)*m, :]


# ==========================================================
# PnP-FBS
# ==========================================================
def pnp_fbs_superresolution(model, im_input, fx, fy, mask,
                            patch_width=64, stride=8, batch_size=500,
                            rho=1.0, maxitr=20):

    H_lr, W_lr = im_input.shape[:2]
    H_hr, W_hr = int(H_lr / fy), int(W_lr / fx)

    y = funcAtranspose(im_input, mask, fx, fy, target_shape=(H_hr, W_hr))
    x = cv2.resize(im_input, (W_hr, H_hr))

    for _ in range(maxitr):

        xold = x.copy()
        xoldhat = funcA(x, mask, fx, fy, target_shape=(H_lr, W_lr))

        grad = funcAtranspose(xoldhat, mask, fx, fy, target_shape=(H_hr, W_hr)) - y
        xtilde = xold - rho * grad

        # ---- Denoising ----
        xtilde_t = torch.from_numpy(xtilde.transpose(2,0,1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            den_out = denoise(model, xtilde_t, patch_width, stride, batch_size)
            den_np = den_out.cpu().numpy().squeeze(0).transpose(1,2,0)

        x = proj(den_np)

    return x.astype(np.float32)


# ==========================================================
# MAIN: Run on a single image
# ==========================================================
if __name__ == "__main__":

    ROOT = os.path.dirname(os.path.abspath(__file__))

    img_path = os.path.join(ROOT, "images", "21077.png")
    model_path = os.path.join(ROOT, "models", "sigma5_RGB.pth")
    save_dir = os.path.join(ROOT, "outputs", "PnP_superresolution_results")
    os.makedirs(save_dir, exist_ok=True)
    # ---- Load GT image ----
    im_orig = cv2.imread(img_path)
    im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


    im_orig = crop_to_multiple(im_orig, 8)
    m, n = im_orig.shape[:2]

    # ---- Gaussian blur ----
    kernel = cv2.getGaussianKernel(9, 1)
    mask = np.outer(kernel, kernel.T)
    r = (mask.shape[0] - 1) // 2

    im_pad = cv2.copyMakeBorder(im_orig, r, r, r, r, borderType=cv2.BORDER_WRAP)
    im_blur = cv2.filter2D(im_pad, -1, mask)[r:r+m, r:r+n, :]   

    # ---- Downsample + noise ----
    K = 2
    im_down = cv2.resize(im_blur, (n//K, m//K), interpolation=cv2.INTER_CUBIC)
    im_noisy = np.clip(im_down + np.random.normal(0, 5/255, im_down.shape), 0, 1)

    # ---- Bicubic baseline ----
    bicubic = cv2.resize(im_down, (n, m), interpolation=cv2.INTER_CUBIC)

    # Save GT + bicubic
    cv2.imwrite(os.path.join(save_dir, "GT.png"), (im_orig*255).astype(np.uint8)[:,:,::-1])
    cv2.imwrite(os.path.join(save_dir, "Bicubic.png"), (bicubic*255).astype(np.uint8)[:,:,::-1])

    # ---- Load OUR model ----
    model = load_ours(model_path)

    # ---- Run PnP-FBS ----
    out = pnp_fbs_superresolution(
        model, im_noisy,
        fx=1/K, fy=1/K,
        mask=mask,
        patch_width=64, stride=8, batch_size=500,
        rho=1.0, maxitr=20
    )

    out = np.clip(out, 0, 1)

    # ---- Compute metrics (EXACT SAME) ----
    ps = psnr_metric(im_orig, out, data_range=1.0)
    ss = ssim_metric(im_orig, out, data_range=1.0, channel_axis=-1)

    print("==============================================")
    print(" RESULTS FOR IMAGE ")
    print("----------------------------------------------")
    print(f"PSNR : {ps:.4f} dB")
    print(f"SSIM : {ss:.6f}")
    print("==============================================")

    cv2.imwrite(os.path.join(save_dir, "OURS.png"),
                (out*255).astype(np.uint8)[:,:,::-1])
