"""A compact, self-contained 2D Fourier Neural Operator (Li et al., 2021) for the operator-learning case.

The spectral convolution is implemented in REAL arithmetic (split real/imag, real learnable weights for each part)
so the whole network exports cleanly to ONNX (opset 18, dynamo) — `torch.complex` + `irfft2` round-trips, and the
complex-weight multiply is a pair of real `einsum`s. The lift/projection are 1x1 convolutions (NOT Linear-on-permuted)
so the ONNX has no shape-baked reshape and accepts any batch. neuraloperator is the reference library; this minimal,
controlled implementation is the genuine method (Fourier layers = spectral conv + 1x1 skip), kept dependency-free and
ONNX-exportable. NEVER import this in the live/Pyodide lane — it is a heavy offline engine.
"""
from __future__ import annotations

import torch


class SpectralConv2d(torch.nn.Module):
    """Truncated spectral convolution: FFT -> keep the lowest (modes1 x modes2) modes, multiply by learnable complex
    weights (real arithmetic) on the two retained corners, inverse FFT."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.m1, self.m2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        sh = (in_ch, out_ch, modes1, modes2)
        self.wr1 = torch.nn.Parameter(scale * torch.randn(*sh))
        self.wi1 = torch.nn.Parameter(scale * torch.randn(*sh))
        self.wr2 = torch.nn.Parameter(scale * torch.randn(*sh))
        self.wi2 = torch.nn.Parameter(scale * torch.randn(*sh))

    @staticmethod
    def _cmul(xr, xi, wr, wi):
        # complex einsum (b,in,x,y),(in,out,x,y)->(b,out,x,y) split into real/imag
        rr = torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi)
        ii = torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr)
        return rr, ii

    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x)
        xr, xi = xf.real, xf.imag
        out_r = torch.zeros(B, self.wr1.shape[1], H, W // 2 + 1, dtype=xr.dtype)
        out_i = torch.zeros_like(out_r)
        r, i = self._cmul(xr[:, :, : self.m1, : self.m2], xi[:, :, : self.m1, : self.m2], self.wr1, self.wi1)
        out_r[:, :, : self.m1, : self.m2] = r
        out_i[:, :, : self.m1, : self.m2] = i
        r, i = self._cmul(xr[:, :, -self.m1 :, : self.m2], xi[:, :, -self.m1 :, : self.m2], self.wr2, self.wi2)
        out_r[:, :, -self.m1 :, : self.m2] = r
        out_i[:, :, -self.m1 :, : self.m2] = i
        return torch.fft.irfft2(torch.complex(out_r, out_i), s=(H, W))


class FNO2d(torch.nn.Module):
    """Lift (1x1 conv) -> L Fourier layers (spectral conv + 1x1 skip, GELU) -> project (1x1 convs). Input channels =
    in_ch (e.g. 3 = coefficient field + x grid + y grid); output = 1 (the solution field)."""

    def __init__(self, in_ch: int = 3, width: int = 20, modes: int = 10, layers: int = 4):
        super().__init__()
        self.fc0 = torch.nn.Conv2d(in_ch, width, 1)
        self.sp = torch.nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.w = torch.nn.ModuleList([torch.nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = torch.nn.Conv2d(width, 64, 1)
        self.fc2 = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.fc0(x)
        for k, (sc, skip) in enumerate(zip(self.sp, self.w)):
            x = sc(x) + skip(x)
            if k < len(self.sp) - 1:
                x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(x)
