# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for FP8 dynamic per-tensor quantization.

The op maps a whole ``[m, n]`` activation tensor into the FP8 E4M3 range using
a single dynamic amax scale, and returns the quantized FP8 values together with
the scalar fp32 scale that dequantizes them.

AMD-runtime semantics (option-b): on gfx950/CDNA4 the FP8 type is OCP
``float8_e4m3fn`` (finite max 448). The scale is ``max(|x|) / 448`` over the
entire tensor; values are rescaled in fp32 and cast to FP8 with
round-to-nearest-even, exactly as the AMD runtime per-tensor quant op
(``get_hip_quant(QuantType.per_Tensor)`` -> ``dynamic_per_tensor_quant``) does.
The returned scale is fp32 of shape ``[1]``.
"""
import torch
import torch.nn as nn

# gfx950 / CDNA4 OCP FP8 (E4M3, finite max 448).
_FP8_DTYPE = torch.float8_e4m3fn


class Model(nn.Module):
    """FP8 dynamic per-tensor quantizer. ``Model()`` takes no hyperparameters."""

    def __init__(self):
        super().__init__()
        self.dtype_max = float(torch.finfo(_FP8_DTYPE).max)

    def forward(self, input):
        x = input.float()
        scale = x.abs().max() / self.dtype_max
        y = (x / scale).to(_FP8_DTYPE)
        return y, scale.view(1).to(torch.float32)


def get_inputs():
    return [torch.randn(128, 4096, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
