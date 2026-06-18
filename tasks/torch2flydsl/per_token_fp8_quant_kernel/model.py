# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for FP8 dynamic per-token quantization.

The op maps each row (token) of a ``[m, n]`` activation tensor into the FP8
E4M3 range using a dynamic per-row amax scale, and returns the quantized FP8
values together with the fp32 scale that dequantizes them.

AMD-runtime semantics (option-b): on gfx950/CDNA4 the FP8 type is OCP
``float8_e4m3fn`` (finite max 448). For each row the scale is
``amax(|x|) / 448``; an all-zero row keeps a scale of 1 to avoid division by
zero. Values are rescaled in fp32 and cast to FP8 with round-to-nearest-even,
exactly as the AMD runtime per-token quant op
(``get_hip_quant(QuantType.per_Token)`` -> ``dynamic_per_token_scaled_quant``)
does. The returned scale is fp32 of shape ``[m, 1]``.
"""
import torch
import torch.nn as nn

# gfx950 / CDNA4 OCP FP8 (E4M3, finite max 448).
_FP8_DTYPE = torch.float8_e4m3fn


class Model(nn.Module):
    """FP8 dynamic per-token quantizer. ``Model()`` takes no hyperparameters."""

    def __init__(self):
        super().__init__()
        self.dtype_max = float(torch.finfo(_FP8_DTYPE).max)

    def forward(self, input):
        x = input.float()
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = amax / self.dtype_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        y = (x / scale).to(_FP8_DTYPE)
        return y, scale.to(torch.float32)


def get_inputs():
    return [torch.randn(128, 4096, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
