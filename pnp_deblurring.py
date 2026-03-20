# ==========================================================
#  IMPORTS
# ==========================================================
import os
import numpy as np
import torch
import cv2
from utils import infer_patchwise_tukey_batched as denoise
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ==========================================================
#  SINGLE IMAGE PATH
# ==========================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(ROOT, "images", "kodim08.png")
save_root = os.path.join(ROOT, "outputs", "PnP_deblur_outputs")
os.makedirs(save_root, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "models", "sigma5_RGB.pth")


# ==========================================================
#  RANDOM SPARSE KERNEL
# ==========================================================
def random_sparse_kernel(ksize=15, sparsity=0.9):
    k = np.random.rand(ksize, ksize).astype(np.float32)
    mask = np.random.rand(ksize, ksize) > sparsity
    k *= mask
    s = k.sum()
    if s == 0:
        k[ksize//2, ksize//2] = 1.0
        s = 1.0
    return k / s

# ==========================================================
#  UTILS
# ==========================================================
def crop_to_multiple(img, m=8):
    H, W = img.shape[:2]
    return img[: (H // m) * m, : (W // m) * m, :]

def proj(im_input, mn, mx):
    return np.clip(im_input, mn, mx)

# ==========================================================
#  FORWARD MODEL: FUNC A AND A^T (FFT VERSION)
# ==========================================================
def funcA(im_input, mask):
    H, W, C = im_input.shape
    H_fft = np.fft.fft2(mask, s=(H, W))
    out = np.zeros_like(im_input)
    for c in range(C):
        X = np.fft.fft2(im_input[..., c])
        Y = np.fft.ifft2(H_fft * X)
        out[..., c] = Y.real
    return out

def funcAtranspose(im_input, mask):
    H, W, C = im_input.shape
    H_fft = np.fft.fft2(mask, s=(H, W))
    out = np.zeros_like(im_input)
    for c in range(C):
        Y = np.fft.fft2(im_input[..., c])
        X = np.fft.ifft2(np.conj(H_fft) * Y)
        out[..., c] = X.real
    return out

# ==========================================================
#  LOAD OUR NONEXPANSIVE MODEL
# ==========================================================
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

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint, strict=True)
model.eval()

print("\nLoaded OUR model.\n")

# ==========================================================
#  PnP-FBS FUNCTION 
# ==========================================================
def pnp_fbs_deblur(model, im_input, mask, **opts):
    rho         = opts.get("rho", 2.0)
    maxitr      = opts.get("maxitr", 30)
    sigma       = opts.get("sigma", 5.0/255.0)
    stride      = opts.get("stride", 8)
    patch_width = opts.get("patch_width", 64)
    batch_size  = opts.get("batch_size", 500)

    y = im_input
    x = funcAtranspose(y, mask)

    print("\nRunning PnP-FBS with OUR model...\n")

    for itr in range(maxitr):
        xold = np.copy(x)

        # Forward-Backward
        xhat  = funcA(x, mask)
        gradx = funcAtranspose(xhat - y, mask)
        xtilde = xold - rho * gradx

        # Denoising
        xt_torch = torch.from_numpy(xtilde.transpose(2,0,1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            x_d = denoise(model, xt_torch, patch_width, stride, batch_size)
            x_d = x_d.cpu().numpy().squeeze(0).transpose(1,2,0)

        x = proj(x_d, 0, 1)

        print(f"Iter {itr+1:02d}/{maxitr}")

    return x

# ==========================================================
#  LOAD IMAGE
# ==========================================================
im_orig = cv2.imread(IMAGE_PATH)
im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
im_orig = crop_to_multiple(im_orig, 8)

print(f"\nLoaded image: {IMAGE_PATH}")
print("Shape:", im_orig.shape)

# ==========================================================
#  APPLY RANDOM SPARSE BLUR + NOISE
# ==========================================================
mask = random_sparse_kernel(ksize=15, sparsity=0.9)

im_blur = funcA(im_orig, mask)

sigma = 5.0/255.0
im_noisy = np.clip(im_blur + np.random.normal(0, sigma, im_blur.shape), 0, 1)

# Save blurred/noisy
cv2.imwrite(os.path.join(save_root, "Blurred.png"), (im_blur[...,::-1]*255).astype(np.uint8))
cv2.imwrite(os.path.join(save_root, "Noisy.png"), (im_noisy[...,::-1]*255).astype(np.uint8))
cv2.imwrite(os.path.join(save_root, "GroundTruth.png"), (im_orig[...,::-1]*255).astype(np.uint8))

# ==========================================================
#  RUN OUR MODEL
# ==========================================================
out = pnp_fbs_deblur(
    model, im_noisy, mask,
    rho=1.0, sigma=sigma, maxitr=30
)

out = np.clip(out, 0, 1)

# Save result
cv2.imwrite(os.path.join(save_root,"OURS.png"), (out[...,::-1]*255).astype(np.uint8))
# ==========================================================
#  COMPUTE FINAL PSNR / SSIM
# ==========================================================
ps = psnr_metric(im_orig, out, data_range=1.0)
ss = ssim_metric(im_orig, out, data_range=1.0, channel_axis=-1)

print("\n====================================")
print(" FINAL RESULTS (kodim08, OUR MODEL)")
print("====================================")
print(f"PSNR: {ps:.4f} dB")
print(f"SSIM: {ss:.4f}")
print("====================================\n")

print("Saved outputs successfully.\n")
