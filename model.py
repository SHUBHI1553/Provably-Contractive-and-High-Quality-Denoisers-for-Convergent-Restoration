import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pywt
import h5py
import random
from pytorch_wavelets import DWTForward, DWTInverse
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- core layers -----
class gradientlayer(nn.Module):
    def __init__(self, grad_in):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(grad_in))

    def forward(self, x, y):
        alpha_clamped = torch.clamp(self.alpha, 0.0, 0.9999)
        return (1 - alpha_clamped) * x + alpha_clamped * y



class proxlayer(nn.Module):
    def __init__(self, prox_in, wavefamily="db4", levels=5):
        super().__init__()
        # ---- learned thresholds for high-frequency subbands ----
        self.lamda1 = nn.Parameter(prox_in[0] * torch.ones((1, 3, 3, 32, 32), dtype=torch.float32))
        self.lamda2 = nn.Parameter(prox_in[1] * torch.ones((1, 3, 3, 16, 16), dtype=torch.float32))
        self.lamda3 = nn.Parameter(prox_in[2] * torch.ones((1, 3, 3, 8, 8), dtype=torch.float32))

        # ---- Softplus mimicking hard clamp ----
        # beta=20 & threshold=20 ? behaves like ReLU; +0.001 offsets clamp floor
        self.softplus = nn.Softplus(beta=20, threshold=20)

        # ---- Wavelet transforms ----
        self.xfwt = DWTForward(J=levels, wave=pywt.Wavelet(wavefamily), mode="periodization").to(device)
        self.xinvfwt = DWTInverse(wave=pywt.Wavelet(wavefamily), mode="periodization").to(device)

        # ---- Shrinkage nonlinearity ----
        self.thresh = nn.ReLU()

    def forward(self, x, y):
        # ---- Softplus ensures strictly positive thresholds ----
        lam1 = 1e-6 + self.softplus(self.lamda1)
        lam2 = 1e-6 + self.softplus(self.lamda2)
        lam3 = 1e-6 + self.softplus(self.lamda3)

        # ---- Wavelet decomposition ----
        (yl, yh) = self.xfwt(x)

        # ---- Apply soft-thresholding to high-frequency bands ----
        yhnew = [
            self.thresh(yh[0] - lam1) - self.thresh(-yh[0] - lam1),
            self.thresh(yh[1] - lam2) - self.thresh(-yh[1] - lam2),
            self.thresh(yh[2] - lam3) - self.thresh(-yh[2] - lam3),
        ]

        # ---- Reconstruct with unchanged lowpass ----
        return self.xinvfwt((yl, yhnew))



class LipschitzConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape, alpha_ref,
                 kernel_size=3, stride=1, padding=1):
        super(LipschitzConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.input_shape = input_shape
        self.alpha_ref = alpha_ref  # linking to gradientlayer.alpha

    def forward(self, x):

        alpha_val = torch.clamp(self.alpha_ref, 0.0, 0.9999)
        scale = 1.0 / (1.0 - alpha_val + 1e-6)
        return scale * self.conv(x)


class LipschitzConv2dPlain(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.input_shape = input_shape

    def forward(self, x):
        return self.conv(x)


class ProxLipBlock(nn.Module):
    """
    Block: LipConv_scaled( Prox( Gradient(z, y) ) )
    """
    def __init__(self, grad_layer, prox_module, lipconv_scaled):
        super().__init__()
        self.grad = grad_layer
        self.prox = prox_module
        self.lipconv = lipconv_scaled

    def forward(self, z, y):
        g = self.grad(z, y)
        p = self.prox(g, y)
        return self.lipconv(p)



class nonexpansivenn_with_conv(nn.Module):
    def __init__(self, prox_in, wavefamilyset, input_shape,
                 grad_in=0.1, levels=5, num_of_layers=10):
        super().__init__()
        self.layers = nn.ModuleList()
        C = 3
        for _ in range(num_of_layers):
            for wave in wavefamilyset:
                g = gradientlayer(grad_in)
                prox = proxlayer(prox_in, wave, levels)
                lip_scaled = LipschitzConv2d(C, C, input_shape, alpha_ref=g.alpha)
                block = ProxLipBlock(g, prox, lip_scaled)

                self.layers.append(block)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out, x)
        return out

