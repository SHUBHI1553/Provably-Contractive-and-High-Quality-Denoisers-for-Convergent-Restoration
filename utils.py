import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F
import itertools
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import time


def overlap_and_add(ps: int, stride: int, device, dtype):
      
    overlap = ps - stride
    alpha = 0.0 if overlap <= 0 else min(1.0, 2.0 * overlap / max(1, (ps - 1)))

    # 1D Tukey
    n = torch.arange(ps, device=device, dtype=dtype)
    w = torch.ones(ps, device=device, dtype=dtype)
    if alpha > 0:
        L = int(round(alpha * (ps - 1) / 2.0))  # taper length per side ≈ overlap
        if L > 0:
            k = torch.arange(L, device=device, dtype=dtype)
            # cosine ramp from 0..L-1 (half-cosine taper)
            taper = 0.5 * (1.0 + torch.cos(torch.pi * (2.0 * k / (alpha * (ps - 1)) - 1.0)))
            w[:L] = taper
            w[-L:] = torch.flip(taper, dims=[0])

   
    w2 = (w[:, None] * w[None, :])
    w2 = w2 / (w2.max() + 1e-12)
    return w2[None, None, ...]  # (1,1,ps,ps)

@torch.no_grad()
def denoising_image(
    model,
    noisy,
    patch_size: int = 64,
    stride: int = 4,
    max_patches_per_batch: int = 128,
):
    """
    Patch-wise denoising with overlap-and-add blending (batched).
    """
    model_was_training = model.training
    model.eval()

    B, C, h, w = noisy.shape
    device, dtype = noisy.device, noisy.dtype
    ps, s = patch_size, stride
    assert ps % s == 0, 

    overlap = ps - s

    # 1) Base pad by overlap on all sides: avoid border artifacts in interior
    base_pad = (overlap, overlap, overlap, overlap)  # (left, right, top, bottom)
    x = F.pad(noisy, base_pad, mode="reflect")
    _, _, H0, W0 = x.shape

    # 2) Tail pad so that tiling grid closes exactly
    #    We want (H_ext - ps) % s == 0 and (W_ext - ps) % s == 0
    tail_h = (s - ((H0 - ps) % s)) % s
    tail_w = (s - ((W0 - ps) % s)) % s
    x = F.pad(x, (0, tail_w, 0, tail_h), mode="reflect")
    _, _, H, W = x.shape

    # 3) COLA Tukey window (1,1,ps,ps), taper length = overlap
    win2d = overlap_and_add(ps, s, device, dtype)  # you already have this

    out = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    weight = torch.zeros((B, 1, H, W), device=device, dtype=dtype)  # scalar weight per pixel

    # 4) Enumerate all patch coordinates on extended canvas
    coords = [
        (i, j)
        for i in range(0, H - ps + 1, s)
        for j in range(0, W - ps + 1, s)
    ]
    num_patches = len(coords)
    if num_patches == 0:
        # degenerate fallback
        y_full = model(x)
        y = y_full[:, :, overlap:overlap + h, overlap:overlap + w].clamp(0, 1)
        if model_was_training:
            model.train()
        return y

    # 5) Process patches in batches
    # Each coord contributes B patches (one per image in batch).
    idx = 0
    while idx < num_patches:
        batch_coords = coords[idx : idx + max_patches_per_batch]
        this_M = len(batch_coords)

        # Collect patches:
        # shape: (this_M * B, C, ps, ps)
        patches = torch.empty(
            (this_M * B, C, ps, ps),
            device=device,
            dtype=dtype,
        )

        # Scatter-add back with Tukey window
        k = 0

        for (ii, jj) in batch_coords:
            # x[:, :, ii:ii+ps, jj:jj+ps] has shape (B, C, ps, ps)
            patches[k : k + B] = x[:, :, ii:ii+ps, jj:jj+ps]
            k += B

        # Run model once on this batch of patches
        den_patches = model(patches)  # (this_M * B, C, ps, ps)

        # Scatter-add back with Tukey window
        k = 0
        for (ii, jj) in batch_coords:
            # Take the next B outputs, reshape to (B,C,ps,ps)
            den_block = den_patches[k : k + B] * win2d  # broadcast win2d over B,C
            # Add contributions
            out[:, :, ii:ii+ps, jj:jj+ps] += den_block
            weight[:, :, ii:ii+ps, jj:jj+ps] += win2d  # same window applied to weights
            k += B

        idx += max_patches_per_batch

    # 6) Crop back to original image region
    top = overlap
    left = overlap
    bot = top + h
    right = left + w

    out_c = out[:, :, top:bot, left:right]
    wgt_c = weight[:, :, top:bot, left:right]

    # For a correct COLA design, wgt_c should be (almost) constant > 0.
    eps = 1e-12
    y = (out_c / (wgt_c + eps)).clamp(0, 1)

    return y


