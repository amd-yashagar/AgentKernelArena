# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Pure-PyTorch reference for the FP8 per-token GEMM ``gemm_a8w8``.

Computes ``out = a @ w.T`` where the activation ``a`` (``[M, K]``) and weight
``w`` (``[N, K]``) are quantized to FP8 (``float8_e4m3fn``) with per-token
scales, matching the AMD runtime per-token FP8 GEMM:

- activation: one FP8 scale per row of ``a`` (``x_scale`` is ``[M, 1]``);
- weight: one FP8 scale per row of ``w`` (``w_scale`` is ``[N, 1]``);
- the GEMM dequantizes each operand (FP8 value times its row scale),
  accumulates in fp32, and truncates the result to bf16.

``quantize_a8w8`` is the single source of the FP8 operands and scales (the
harness reuses it to drive the real AMD runtime op), so the reference and the
hardware kernel operate on byte-identical inputs and differ only by fp32
accumulation order.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# e4m3fn saturation magnitude used as the per-token scale divisor on gfx950.
_FP8_MAX = 448.0


def _quantize_rowwise(t):
    """Per-row FP8 quantization of a 2-D tensor.

    Returns the FP8 codes (``float8_e4m3fn``) and the fp32 row scales
    (``[rows, 1]``) such that ``t ~= codes.float() * scale``."""
    tf = t.float()
    amax = tf.abs().amax(dim=-1, keepdim=True)
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    q = (tf / scale).to(torch.float8_e4m3fn)
    return q, scale


def quantize_a8w8(a, w):
    """Quantize a high-precision activation/weight pair to the per-token FP8
    operands the deployed GEMM consumes.

    Returns ``(x_fp8, x_scale, w_fp8, w_scale)`` with the layouts the AMD
    runtime per-token FP8 GEMM expects (``x_scale`` is ``[M, 1]``, ``w_scale``
    is ``[N, 1]``)."""
    x_fp8, x_scale = _quantize_rowwise(a)
    w_fp8, w_scale = _quantize_rowwise(w)
    return x_fp8, x_scale, w_fp8, w_scale


def _dequant_matmul(x_fp8, x_scale, w_fp8, w_scale):
    """Dequantize the FP8 per-token operands and run the fp32 GEMM."""
    x = x_fp8.float() * x_scale
    w = w_fp8.float() * w_scale
    return F.linear(x, w)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, w):
        x_fp8, x_scale, w_fp8, w_scale = quantize_a8w8(a, w)
        out = _dequant_matmul(x_fp8, x_scale, w_fp8, w_scale)
        return out.to(torch.bfloat16)


def get_inputs():
    # Representative FP8 a8w8 shape (M, N, K) = (128, 1280, 8192); the harness
    # sweeps more real shapes from configs/a8w8_untuned_gemm.csv.
    m, n, k = 128, 1280, 8192
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    return [a, w]


def get_init_inputs():
    return []
