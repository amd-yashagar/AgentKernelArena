# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""PyTorch reference (KernelBench format) for the QUANTIZED a4w4 fused MoE.

This is an apple-to-apple translation source for the FlyDSL a4w4 kernel: the
FlyDSL MoE kernels are mxfp4/a4w4 only (no fully-bf16 MoE GEMM exists), so the
PyTorch reference faithfully EMULATES THE SAME QUANTIZED COMPUTATION:

  1. quantize activations + expert weights to MXFP4 (`float4_e2m1fn_x2`) with
     e8m0 `per_1x32` block scales, using the same scheme AITER uses
     (`get_torch_quant(QuantType.per_1x32)`);
  2. grouped gate/up GEMM over DEQUANTIZED values with fp32 accumulation;
  3. silu(gate) * up;
  4. RE-QUANTIZE the stage-1 result to mxfp4 (exactly as the kernel does
     before stage2);
  5. down GEMM (fp32 accumulate over dequantized values);
  6. weighted top-k combine; output bf16.

Steps 2/3 and 5/6 are AITER's canonical pure-torch quantized references
`torch_moe_stage1` / `torch_moe_stage2` (they dequantize the mxfp4 tensors and
do fp32 matmuls). Routing (softmax top-k, renormalized) is computed by
`route_topk`, which the harness reuses so the reference and the kernel select
the same experts.
"""
import torch
import torch.nn as nn


def route_topk(logits, topk):
    """Softmax router + top-k with renormalized weights. Shared by the harness."""
    gate = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(gate, topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), ids.to(torch.int32)


class Model(nn.Module):
    def __init__(self, model_dim, inter_dim, experts, topk, activation="silu"):
        super().__init__()
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.topk = topk
        self.activation = activation
        self.gate = nn.Linear(model_dim, experts, bias=False).to(torch.bfloat16)
        self.w1 = nn.Parameter(
            (torch.randn(experts, 2 * inter_dim, model_dim) / 10).to(torch.bfloat16)
        )
        self.w2 = nn.Parameter(
            (torch.randn(experts, model_dim, inter_dim) / 10).to(torch.bfloat16)
        )

    def forward(self, hidden_states):
        import aiter
        from aiter import QuantType, dtypes, ActivationType
        from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

        E, I = self.experts, self.inter_dim
        act = ActivationType.Silu if self.activation == "silu" else ActivationType.Gelu
        tq = aiter.get_torch_quant(QuantType.per_1x32)

        # routing (shared with the kernel via the harness)
        logits = self.gate(hidden_states)
        topk_weights, topk_ids = route_topk(logits, self.topk)

        # (1) quantize activations + weights to mxfp4 + e8m0 per_1x32 scales
        a1_qt, a1_scale = tq(hidden_states, quant_dtype=dtypes.fp4x2)
        w1_qt, w1_scale = tq(self.w1, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = tq(self.w2, quant_dtype=dtypes.fp4x2)
        w1_qt = w1_qt.view(E, 2 * I, self.model_dim // 2)
        w2_qt = w2_qt.view(E, self.model_dim, I // 2)

        # (2,3) stage1: dequant gate/up GEMM (fp32 accum) + silu(gate) * up
        stage1 = torch_moe_stage1(
            a1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
            dtype=torch.bfloat16, activation=act, quant_type=QuantType.per_1x32,
            a1_scale=a1_scale, w1_scale=w1_scale, doweight=False,
        )

        # (4) re-quantize the stage-1 result to mxfp4 (as the kernel does)
        a2_qt, a2_scale = tq(stage1.view(-1, I), quant_dtype=dtypes.fp4x2)
        a2_qt = a2_qt.view(hidden_states.shape[0], self.topk, -1)

        # (5,6) stage2: dequant down GEMM (fp32 accum) + weighted top-k combine
        out = torch_moe_stage2(
            a2_qt, w1_qt, w2_qt, topk_weights, topk_ids,
            quant_type=QuantType.per_1x32, w2_scale=w2_scale, a2_scale=a2_scale,
            doweight=True,
        )
        return out.to(torch.bfloat16)


def get_inputs():
    # Default representative shape: tokens=16, model_dim=7168 (DeepSeek-V3 row).
    return [torch.randn(16, 7168, dtype=torch.bfloat16)]


def get_init_inputs():
    # Flat positional args for Model(*get_init_inputs()), matching
    # Model.__init__(model_dim, inter_dim, experts, topk, activation="silu").
    # dsv3 default (D=7168, I=256, E=257, topk=9); activation stays "silu".
    return [7168, 256, 257, 9]
