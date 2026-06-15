# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference (KernelBench format) for half-precision GEMM.

out = a @ b.T  with fp32 accumulation, where `a` is [M, K] and `b` is [N, K].
This matches the FlyDSL `flydsl_hgemm(a, b)` operand layout (b stored [N, K]).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # fp32 accumulation, result cast back to the input dtype.
        return torch.matmul(a.float(), b.float().transpose(-1, -2)).to(a.dtype)


def get_inputs():
    # Default representative shape (M, N, K) = (256, 256, 5120) from
    # aiter/configs/bf16_untuned_gemm.csv. The harness sweeps more real shapes.
    m, n, k = 256, 256, 5120
    a = torch.rand(m, k, dtype=torch.bfloat16)
    b = torch.rand(n, k, dtype=torch.bfloat16)
    return [a, b]


def get_init_inputs():
    return []
