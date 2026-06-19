# Triton kernel surface catalog — for `triton2flydsl` task generation

Inventory of the **standalone Triton kernels** that are candidates for
`triton2flydsl` tasks (a standalone Triton+torch kernel is the SOURCE/reference;
FlyDSL is GEAK's target, added later — see the sibling-suite callout in
[`TORCH2FLYDSL_CONVENTIONS.md`](TORCH2FLYDSL_CONVENTIONS.md) and the suite design
in [`HANDOFF_GFX942.md`](HANDOFF_GFX942.md) §A). Each task ships `<name>.py`
(Triton source, faithful to upstream, **no torch reference file, no `kernel.py`**)
+ `test_kernel_harness.py` + `config.yaml` (`task_type: triton2flydsl`); the
harness **runs the Triton kernel** and validates finiteness (+ closeness to a
trivial inline torch reference where one exists).

This is the `triton2flydsl` counterpart to
[`AITER_OPS_CATALOG.md`](AITER_OPS_CATALOG.md) (which drives the `torch2flydsl`
PyTorch-`Model`-source suite). Rows here are reconciled against that catalog's
Triton not-yet / skip lists so nothing is duplicated or proposed as a poor fit.

- **AITER source**: `/workspaces/meta/aiter`, HEAD `e773193041689602e5431e110ef39db0390e7015`.
- **generative-recommenders source**: `/workspaces/meta/generative-recommenders`
  (the `generative_recommenders/ops/triton/*` kernels), pinned at commit
  `456ff31a76d7a96aa18f0723436897150adb8734` (branch `main`); origin of the
  `jagged_dense_bmm_broadcast_add` task and the 3 new GR tasks.
- **SGLang source**: `sglang` (`python/sglang/srt/layers/.../*` Triton kernels),
  pinned at commit `9bb9d17e1ab26d50b91e5672b7ffcac800faafc7` (branch `main`);
  origin of the 5 GDN / fused-MoE SGLang tasks.
- **Target arch**: gfx942 (CDNA3, MI300X — this node) + gfx950 (CDNA4). flydsl 0.2.1.
- **Existing tasks**: `tasks/triton2flydsl/` (aiter 26 + generative-recommenders 4
  + SGLang 21 = **51**). aiter = 9 prior + 5 Batch 1 + 5 Batch 2 + 7 Batch 3;
  generative-recommenders = `jagged_dense_bmm_broadcast_add` (prior) + 3 new;
  SGLang = 5 GDN / fused-MoE (prior) + 5 GDN chunk pipeline (Batch S1) + 11
  Batch S2–S4 (RadixAttention backend + elementwise/MoE/mrope + linear-attn/SSD +
  dsv4 FP4 quant).

---

## Legend

| Field | Meaning |
|---|---|
| **op** | Op name + the standalone entry (`module:function`). |
| **source** | aiter `ops/triton/<path>` (or generative-recommenders path). Device kernels live under `_triton_kernels/`. |
| **standalone?** | Can it be extracted to **triton + torch only** (the gating criterion)? `clean` = yes, minimal inlining; `inline` = yes after inlining util helpers (arch/kernel_repr/pid); `heavy` = pulls disk configs / many deps; `NO` = needs gluon / aiter device deps. |
| **upstream test** | `op_tests/triton_tests/<file>` reference to mirror (torch ref where present). |
| **AMD dtype** | dtypes the op runs. fp8 on gfx942 = `e4m3fnuz`; on gfx950 = `e4m3fn`. |
| **arch** | **942** = runs on gfx942/gfx950; **950-only** = FP4 / MX scaled-dot (`tl.dot_scaled`/`DotScaleOp`) — CDNA3 has no scaled-MFMA (see HANDOFF §D). |
| **models** | Production models that exercise it (DS=DeepSeek-V3/V4, GPT-OSS, Llama, Qwen3, GLM, Kimi, MiniMax, GR=generative-recommenders). |
| **diff** | S simple · M quant/fused · L stateful/metadata/attention · XL impractical. |
| **status** | `DONE` (+ dir) / `addable` (queued) / `skip` (poor fit + reason). |

---

## Summary counts

Totals span all three sources (aiter + generative-recommenders + SGLang).

| Bucket | aiter | GR | SGLang | **Total** |
|---|--:|--:|--:|--:|
| **DONE — triton2flydsl** | 26 | 4 | 21 | **51** |
| **addable — gfx942-runnable** | ~28 | 8 | 1 | **~37** |
| **addable — gfx950-only** (FP4 / MX scaled-dot) | ~16 | 0 | 0 | **~16** |
| **skip — poor fit** (gluon / bwd / multi-GPU / infra / niche) | ~22 | 5 | families* | **~27+** |

\* SGLang skips are recorded as families (duplicates-of-aiter + EP/LoRA/KV-meta/
spec-decode/grammar/diffusion/SM100/backward) rather than a single count — see the
SGLang verdict. Per-source breakdowns: aiter §1–§8, generative-recommenders §GR,
SGLang §SG.

- **DONE by source:** aiter 26 (9 prior + 5 Batch 1 + 5 Batch 2 + 7 Batch 3),
  generative-recommenders 4 (1 prior + 3 new), SGLang 21 (5 prior + 5 Batch S1 +
  11 Batch S2–S4).

Batch S2–S4 added **11** (all gfx942, all `--correctness` exit 0). RadixAttention
native backend (S2): `merge_state` (flash-decoding combine, 7/7), `prefill_attention`
(`context_attention_fwd` varlen flash prefill, 7/7), `extend_attention`
(prefill-with-KV-cache two-stage, incl. MLA split-head, 7/7), `decode_attention`
(two-stage flash-decoding MHA+GQA+MQA, 6/6). Elementwise/MoE/mrope (S3):
`fused_dual_residual_rmsnorm` (8/8), `experts_combine` (8/8),
`triton_mrope_fused` (Qwen2-VL/2.5-VL sectioned M-RoPE, neox+gptj+interleaved, 8/8),
`fused_moe_router` (softcapped top-k, cudacore+tensorcore, 8/8). Linear-attn/SSD
(S3): `lightning_attn` (MiniMax/Bailing decode recurrence, 6/6),
`ssd_chunk_state` (Mamba2 chunk-state fwd, 6/6). dsv4 FP4 (S4):
`dsv4_fp4_indexer` (DeepSeek-V4 FP4 e2m1+ue8m0 indexer quant + paged cache store,
bit-exact, 6/6) — **reclassified gfx942** (pure bitwise quant, no `tl.dot_scaled`).
The attention/SSD bf16 tasks use the convention's documented dual gate
(isclose 1e-2 over ≥99.9% OR normalized worst-element ≤1e-2); the fp32 paths are tight.

Batch 1 added **5** (all gfx942, all `--correctness` exit 0): `gemm_a16w16`,
`rmsnorm`, `layernorm`, `fused_add_rmsnorm`, `softmax`.

Batch 2 added **5** (all gfx942, all `--correctness` exit 0): `rope_fwd`,
`gemm_a8w8_blockscale`, `gemm_a8w8`, `fused_silu_mul`, `fused_clamp_act_mul`. The
two fp8 GEMMs use the arch-matched fp8 e4m3 dtype (e4m3fnuz on gfx942) for both
inputs and the torch reference at the tight upstream gate.

Batch 3 added **7** (Batch 3 fully cleared). Sub-batch 1 (**4**):
`batched_gemm_bf16` (gfx942, 14/14), `batched_gemm_a8w8` (gfx942, int8 per upstream
test, 14/14), `ff_a16w16` (gfx942, ungated FFN, 20/20 shape×activation) — all
`--correctness` exit 0 — plus `gemm_afp8wfp8` (**gfx950-only skip-guard**: uses
`tl.dot_scaled`; built + ready for a gfx950 node, harness prints SKIPPED / exit 0 on
gfx942). Sub-batch 2 (**3**, all gfx942, `--correctness` exit 0):
`dynamic_quant_fp8` (static + dyn per-tensor + dyn per-token fp8/int8, 48/48),
`dynamic_mxfp8_quant` (per-1x32 e8m0, bit-exact scales, 10/10), and
`gemm_a16w8_blockscale` (a16 act × w8 128×128 blockscale, `tl.dot`, 5/5).

---

## Built-task index (`tasks/triton2flydsl/` — 51 dirs)

### aiter (26)

**Prior (9):** `aiter/mha`, `aiter/mla`, `aiter/fav3_sage`,
`aiter/fav3_sage_mxfp4` (gfx950-only), `aiter/fp8_mqa_logits`,
`aiter/unified_attention`, `aiter/unified_attention_sparse_mla`,
`aiter/moe_fused_gemm`, `aiter/moe_routing_sigmoid_top1`.

**Batch 1 (5, gfx942):** `aiter/gemm_a16w16`, `aiter/rmsnorm`,
`aiter/layernorm`, `aiter/fused_add_rmsnorm`, `aiter/softmax`.

**Batch 2 (5, gfx942):** `aiter/rope_fwd`, `aiter/gemm_a8w8_blockscale`,
`aiter/gemm_a8w8`, `aiter/fused_silu_mul`, `aiter/fused_clamp_act_mul`.

**Batch 3 (7):** sub-batch 1 — `aiter/batched_gemm_bf16`, `aiter/batched_gemm_a8w8`,
`aiter/ff_a16w16` (all gfx942), `aiter/gemm_afp8wfp8` (gfx950-only skip-guard);
sub-batch 2 — `aiter/dynamic_quant_fp8`, `aiter/dynamic_mxfp8_quant`,
`aiter/gemm_a16w8_blockscale` (all gfx942).

### generative-recommenders (4, all gfx942)

**Prior (1):** `generative_recommenders/jagged_dense_bmm_broadcast_add`.

**New (3):** `generative_recommenders/jagged_dense_broadcast_add`,
`generative_recommenders/layer_norm`, `generative_recommenders/swiglu`.

### SGLang (21, all gfx942)

**Prior (5):** `sglang/gdn_chunk_fwd_h`, `sglang/gdn_chunk_fwd_o`,
`sglang/gdn_fused_recurrent_decode`, `sglang/gdn_l2norm_fwd`,
`sglang/sglang_fused_moe`.

**Batch S1 (5, finishes the GDN/FLA chunk pipeline):**
`sglang/fused_gdn_gating`, `sglang/chunk_local_cumsum`,
`sglang/chunk_scaled_dot_kkt_fwd`, `sglang/fused_norm_gate`,
`sglang/wy_fast`.

**Batch S2–S4 (11):** RadixAttention native backend —
`sglang/merge_state`, `sglang/prefill_attention`, `sglang/extend_attention`,
`sglang/decode_attention`; elementwise/MoE/mrope —
`sglang/fused_dual_residual_rmsnorm`, `sglang/experts_combine`,
`sglang/triton_mrope_fused`, `sglang/fused_moe_router`; linear-attn/SSD —
`sglang/lightning_attn` (decode), `sglang/ssd_chunk_state` (Mamba2 chunk-state);
dsv4 FP4 — `sglang/dsv4_fp4_indexer` (gfx942, bit-exact).

---

## 1. Normalization

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **rmsnorm** `rmsnorm:rms_norm` | `normalization/rmsnorm.py` | clean | `normalization/test_rmsnorm.py:torch_rmsnorm` | bf16/fp16/fp32, fp32 reduce | 942 | all | S | **DONE** `aiter/rmsnorm` |
| **fused_add_rmsnorm** `rmsnorm:rmsnorm2d_fwd_with_add` | `normalization/rmsnorm.py` | clean | `test_rmsnorm.py:test_fused_add_rmsnorm` | bf16/fp16 | 942 | all | S | **DONE** `aiter/fused_add_rmsnorm` |
| **layernorm** `norm:layer_norm` | `normalization/norm.py` | clean | `normalization/test_layernorm.py:run_torch` (F.layer_norm) | bf16/fp16/fp32 | 942 | GPT-style | S | **DONE** `aiter/layernorm` |
| **fused_add_rmsnorm_pad** `fused_add_rmsnorm_pad` | `normalization/fused_add_rmsnorm_pad.py` | clean | `normalization/test_fused_add_rmsnorm_pad.py` | bf16 | 942 | padded hidden dims | S | addable |
| **fused_rmsnorm_add** (alt fused) | `normalization/fused_rmsnorm_add.py` | clean | `normalization/test_fused_rmsnorm_add.py:run_torch` | bf16 | 942 | all | S | addable (overlaps `fused_add_rmsnorm`; low-pri) |
| **layernorm_fused_add** `norm:layernorm2d_fwd_with_add` | `normalization/norm.py` | clean | `test_layernorm.py:test_fused_add_layernorm` | bf16 | 942 | GPT-style | S | addable (overlaps `layernorm`) |
| **rmsnorm/layernorm + dynamic/smooth quant** | `normalization/{rmsnorm,norm}.py` | inline | `test_rmsnorm.py`/`test_layernorm.py` quant cases | int8/fp8 (e4m3fnuz) | 942 | quantized serving | M | addable |

## 2. Activation

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **fused_silu_mul** `activation:fused_silu_mul` | `activation.py` | clean | `fusions/test_fused_silu_mul.py` | bf16/fp16 | 942 | all MoE/FFN | S | **DONE** `aiter/fused_silu_mul` |
| **fused_clamp_act_mul** `fusions/fused_clamp_act_mul.py` | `fusions/fused_clamp_act_mul.py` | clean | `fusions/test_fused_clamp_act_mul.py` | bf16 | 942 | **GPT-OSS** (clamped swiglu) | S/M | **DONE** `aiter/fused_clamp_act_mul` (non-quant path) |
| **act_mul_and_fp8_group_quant** `activation:act_mul_and_fp8_group_quant` | `activation.py` | inline (drop `aiter.dtypes`) | `quant/test_fused_fp8_quant.py` | fp8 group (e4m3fnuz) | 942 | MoE pre-quant | M | addable |
| **act_mul_and_mxfp4_quant** `activation:act_mul_and_mxfp4_quant` | `activation.py` | inline | `quant/test_fused_mxfp4_quant.py` | mxfp4 (e2m1+e8m0) | 950-only | fp4 MoE | M | addable (950) |

## 3. GEMM (basic / batched / feed-forward / fused)

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **gemm_a16w16** `gemm.basic.gemm_a16w16:gemm_a16w16` | `gemm/basic/gemm_a16w16.py` | heavy→inline (static cfg, NUM_KSPLIT=1) | `gemm/basic/test_gemm_a16w16.py` (F.linear) | bf16/fp16 | 942 | all dense | M | **DONE** `aiter/gemm_a16w16` |
| **gemm_a16w16_atomic** `gemm_a16w16_atomic` | `gemm/basic/gemm_a16w16_atomic.py` | inline | `test_gemm_a16w16.py:test_*_atomic` | bf16 (fp32 atomic) | 942 | dense | M | addable (overlaps a16w16) |
| **gemm_a16w16_gated** `gemm_a16w16_gated` | `gemm/basic/gemm_a16w16_gated.py` | inline | `test_gemm_a16w16_gated.py` | bf16 | 942 | gated FFN | M | addable |
| **gemm_a8w8** (fp8 per-token/tensor) | `gemm/basic/gemm_a8w8.py` | inline | `gemm/basic/test_gemm_a8w8.py:run_torch` | fp8 (e4m3fnuz) | 942 | Llama/Qwen fp8 | M | **DONE** `aiter/gemm_a8w8` |
| **gemm_a8w8_blockscale** (fp8 128×128) | `gemm/basic/gemm_a8w8_blockscale.py` | inline | `test_gemm_a8w8_blockscale.py:run_torch` | fp8 blockscale | 942 | **DeepSeek-V3**, Qwen3 | M | **DONE** `aiter/gemm_a8w8_blockscale` |
| **gemm_a8w8_per_token_scale** | `gemm/basic/gemm_a8w8_per_token_scale.py` | inline | `test_gemm_a8w8_per_token_scale.py` | fp8 per-token | 942 | fp8 dense | M | addable |
| **gemm_a16w8_blockscale** | `gemm/basic/gemm_a16w8_blockscale.py` | inline | `test_gemm_a16w8_blockscale.py` | a16 act, w8 blockscale | 942 | w8 inference | M | **DONE** `aiter/gemm_a16w8_blockscale` (non-prequant path; uses `tl.dot`) |
| **gemm_afp8wfp8** | `gemm/basic/gemm_afp8wfp8.py` | inline | `test_gemm_afp8wfp8.py` | MXFP8 act + fp8 weight (e8m0 scales) | **950-only** | fp8 dense | M | **DONE** `aiter/gemm_afp8wfp8` (950-only skip-guard; uses `tl.dot_scaled`) |
| **gemm_a16wfp4 / a8wfp4 / afp4wfp4 / afp4wfp4_pre_quant_atomic** | `gemm/basic/gemm_a*fp4*.py` | inline | `gemm/basic/test_gemm_a*fp4*.py` | FP4 (`tl.dot_scaled`) | **950-only** | DS/GPT-OSS fp4 | M | addable (950) |
| **batched_gemm_bf16** | `gemm/batched/batched_gemm_bf16.py` | inline | `gemm/batched/test_batched_gemm_bf16.py` | bf16 | 942 | batched proj | M | **DONE** `aiter/batched_gemm_bf16` |
| **batched_gemm_a8w8 (+ group-prequant)** | `gemm/batched/batched_gemm_a8w8*.py` | inline | `gemm/batched/test_batched_gemm_a8w8*.py` | int8/fp8 (per-row/col scale) | 942 | batched fp8 | M | **DONE** `aiter/batched_gemm_a8w8` (int8 per upstream test; group-prequant variant still addable) |
| **batched_gemm_a16wfp4 / afp4wfp4 (+pre_quant)** | `gemm/batched/batched_gemm_*fp4*.py` | inline | `gemm/batched/test_batched_gemm_*fp4*.py` | FP4 | **950-only** | fp4 batched | M | addable (950) |
| **ff_a16w16 (+fused_gated/ungated)** | `gemm/feed_forward/ff_a16w16*.py` | inline | `gemm/feed_forward/test_ff_a16w16*.py` | bf16 | 942 | FFN | M | **DONE** `aiter/ff_a16w16` (ungated `ff_a16w16_nogate` path; fused_gated/ungated kernels still addable) |
| **fused_gemm_a16w16_quant_x** | `gemm/fused/fused_gemm_a16w16_quant_x.py` | inline | `gemm/fused/test_fused_gemm_a16w16_quant_x.py` | bf16→fp8 | 942 | fp8 pre-quant FFN | M | addable |
| **fused_gemm_a8w8_blockscale_{a16w16,mul_add,split_cat}** | `gemm/fused/fused_gemm_a8w8_blockscale_*.py` | inline | `gemm/fused/test_fused_gemm_a8w8_blockscale_*.py` | fp8 blockscale fused | 942 | DS/Qwen fp8 | M | addable |
| **fused_gemm_afp4wfp4_{a16w16,mul_add,split_cat}** | `gemm/fused/fused_gemm_afp4wfp4_*.py` | inline | `gemm/fused/test_fused_gemm_afp4wfp4_*.py` | FP4 fused | **950-only** | fp4 FFN | M | addable (950) |

## 4. MoE (GEMM + routing/sort)

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **fused_moe** `moe_op:fused_moe` | `moe/moe_op.py` | inline | `moe/test_moe.py` | bf16/fp16 (+fp8/int8 paths) | 942 | all MoE | M | **DONE** `aiter/moe_fused_gemm` |
| **routing_sigmoid_top1** | `moe/moe_routing_sigmoid_top1_fused.py` | inline | `moe/test_moe_routing_sigmoid_top1_fused.py` | fp16 | 942 | GPT-OSS-ish | S | **DONE** `aiter/moe_routing_sigmoid_top1` |
| **moe_align_block_size** | `moe/moe_align_block_size.py` | clean | `moe/test_moe_align_block_size.py` | int (counting sort) | 942 | all MoE | M | addable (already inlined in moe_fused_gemm; standalone dup) |
| **moe_op_gemm_a8w8 (+_blockscale)** | `moe/moe_op_gemm_a8w8*.py` | inline | `moe/test_moe_gemm_a8w8*.py` | fp8 (e4m3fnuz) | 942 | fp8 MoE | M | addable |
| **moe_op_gemm_int8_smoothquant** | `moe/moe_op_gemm_int8_smoothquant.py` | inline | `moe/test_moe_gemm_int8_smoothquant.py` | int8 | 942 | int8 MoE | M | addable |
| **moe_op_silu_fused / moe_op_gelu** | `moe/moe_op_silu_fused.py`, `moe_op_gelu.py` | inline | `moe/test_moe.py` (act paths) | bf16 | 942 | MoE FFN | M | addable |
| **moe routing / topk** | `moe/moe_routing/{routing,topk}.py` | inline | `moe/test_moe_routing.py` | fp32 gate | 942 | all MoE | M | addable |
| **moe_op_gemm_a16w4 / a4w4 / mxfp4 / mxfp4_silu_fused** | `moe/moe_op_gemm_a4w4.py`, `moe_op_mxfp4*.py` | inline | `moe/test_moe_gemm_a4w4.py`, `test_moe_mx.py` | FP4 (`tl.dot_scaled`) | **950-only** | DS/Kimi/GPT-OSS fp4 | M | addable (950) |
| **moe `reduce` / `bitmatrix`** | `moe/reduce.py`, `moe/moe_routing/bitmatrix.py` | — | — | helper | — | — | — | **skip** (infra helper) |

## 5. RoPE

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **rope_fwd (neox/gptj, sbhd)** `rope:rope_fwd` | `rope/rope.py` | inline (sbhd fwd kernel + rotate helpers) | `rope/test_rope.py:ref_rope_sbhd_fwd` | bf16 | 942 | all | M | **DONE** `aiter/rope_fwd` (sbhd non-cached fwd) |
| **fused_qkv_split_qk_rope** | `rope/fused_qkv_split_qk_rope.py` | inline | `rope/test_fused_qkv_split_qk_rope.py:run_torch` | bf16 | 942 | serving | M | addable |
| **fused_qkv_split_qk_norm_rope_cache** | `rope/fused_qkv_split_qk_norm_rope_cache.py` | inline | (fused; norm+rope+cache) | bf16/fp8 | 942 | serving | L | addable |

## 6. Quant (standalone)

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **dynamic per-token / per-tensor fp8 quant** `quant.py` | `quant/quant.py` | inline | `quant/test_quant.py` | fp8 (e4m3fnuz) / int8 | 942 | all fp8 | S | **DONE** `aiter/dynamic_quant_fp8` (static + dyn per-tensor + dyn per-token) |
| **dynamic_mxfp8_quant** `quant.py:dynamic_mxfp8_quant` | `quant/quant.py` | inline | `quant/test_quant_mxfp8.py:torch_mxfp8_quant_from_fp32` | mxfp8 (e4m3+e8m0) | 942 | fp8 | S/M | **DONE** `aiter/dynamic_mxfp8_quant` (bit-exact scales) |
| **fused_fp8_quant** `fused_fp8_quant.py` | `quant/fused_fp8_quant.py` | inline | `quant/test_fused_fp8_quant.py:run_torch_*` | fp8 group | 942 | MoE pre-quant | M | addable |
| **fused_mxfp8_quant** `fused_mxfp8_quant.py` | `quant/fused_mxfp8_quant.py` | inline | `quant/test_quant_mxfp8.py` | mxfp8 fused | 942 | fp8 | M | addable |
| **fused_rms_gated_fp8_group_quant** | `quant/fused_*` / `fusions` | inline | `quant/test_fused_rms_gated_fp8_group_quant.py:ref_rmsnorm_quant` | fp8 group | 942 | DS/Qwen fp8 | M | addable |
| **dynamic_mxfp4 / nvfp4 quant** `quant.py` | `quant/quant.py` | inline | `quant/test_quant_mxfp4.py` | FP4 (e2m1+e8m0/fp8) | **950-only** | fp4 | M | addable (950) |
| **fused_mxfp4_quant** `fused_mxfp4_quant.py` | `quant/fused_mxfp4_quant.py` | inline | `quant/test_fused_mxfp4_quant.py` | mxfp4 fused | **950-only** | fp4 | M | addable (950) |
| **sage_attention_quant_wrappers** | `quant/sage_attention_quant_wrappers.py` | — | (used by fav3_sage) | — | — | — | — | **skip** (used by DONE `fav3_sage`) |

## 7. Attention (beyond the 6 DONE)

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **mha (bf16)** | `attention/mha.py` | inline | `attention/test_mha.py` | bf16 | 942 | all dense | M | **DONE** `aiter/mha` |
| **mla** | `attention/mla.py` | inline | `attention/test_mla.py` | bf16 | 942 | **DeepSeek** | L | **DONE** `aiter/mla` |
| **unified_attention (+sparse_mla)** | `attention/unified_attention*.py` | inline | `attention/test_unified_attention*.py` | bf16 | 942 | vLLM-style/DS | L | **DONE** `aiter/unified_attention`, `aiter/unified_attention_sparse_mla` |
| **fav3_sage (+mxfp4)** | `attention/fav3_sage*.py` | inline | `attention/test_fav3_sage.py` | int8/fp8 (mxfp4) | 942 (mxfp4=950) | long-ctx | XL | **DONE** `aiter/fav3_sage` (+`fav3_sage_mxfp4` 950-only) |
| **fp8_mqa_logits** | `attention/fp8_mqa_logits.py` | inline | `attention/test_fp8_mqa_logits.py:ref_fp8_mqa_logits` | fp8 | 942 | DS indexer | L | **DONE** `aiter/fp8_mqa_logits` |
| **mha_with_sink** | `attention/mha.py` (sink path) | inline | `attention/test_mha_with_sink.py` | bf16 + attention sink | 942 | **GPT-OSS** | M/L | addable (high value) |
| **mha_v3 (fp8)** | `attention/mha_v3.py` | inline | `attention/test_mha_v3.py`, `test_mha_fp8.py` | fp8 (e4m3fnuz) | 942 | fp8 inference | M/L | addable (overlaps mha) |
| **extend_attention** | `attention/extend_attention.py` | inline | `attention/test_extend_attention.py` | bf16 | 942 | serving prefix-extend | L | addable |
| **prefill_attention** | `attention/prefill_attention.py` | inline | `attention/test_prefill_attention.py` | bf16 | 942 | serving prefill | L | addable |
| **pa_decode** | `attention/pa_decode.py` | inline | `attention/test_pa_decode.py` | bf16/fp8 KV, paged | 942 | all serving | L | addable |
| **pa_prefill / chunked_pa_prefill** | `attention/{pa_prefill,chunked_pa_prefill}.py` | inline | `attention/test_{pa_prefill,chunked_pa_prefill}.py` | bf16 paged | 942 | serving | L | addable |
| **pa_decode_sparse** | `attention/pa_decode_sparse.py` | inline | `attention/test_pa_decode_sparse.py` | bf16 sparse paged | 942 | sparse serving | L | addable |
| **hstu_attention** | `attention/hstu_attention.py` | inline | `attention/test_hstu_attn.py` | bf16 | 942 | GR / ranking | L | addable |
| **lean_atten / lean_atten_paged** | `attention/lean_atten*.py` | inline | `attention/test_la*.py` | bf16 | 942 | long-ctx | L | addable |
| **mla_decode / mla_decode_rope** | `attention/mla_decode*.py` | inline | `attention/test_mla_decode_rope.py` | bf16 | 942 | DS decode | L | addable (overlaps `mla`) |
| **mha_with_pe / pod_attention** | `attention/{mha.py,pod_attention.py}` | inline | `attention/test_mha_with_pe.py` | bf16 | 942 | niche | L | addable (low-pri) |
| **mha_fused_bwd / mha_onekernel_bwd** | `attention/mha_*_bwd.py` | — | `test_mha*` bwd | bf16 grad | — | training | XL | **skip** (backward/training) |

## 8. Conv / linear-attn / fusions / other

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **causal_conv1d (+update_single_token)** | `conv/causal_conv1d*.py` | inline | `conv/test_causal_conv1d*.py:causal_conv1d_ref` | bf16/fp16 | 942 | Mamba/hybrid (SSM) | M | addable |
| **gated_delta_rule** | `gated_delta_net/gated_delta_rule.py` | inline | `test_gated_delta_rule` (recurrent/chunk ref) | bf16 | 942 | **Qwen3-Next**, MiniMax | L | addable (stateful recurrence) |
| **fused_mul_add** | `fusions/fused_mul_add.py` | clean | `fusions/test_fused_mul_add.py` | bf16 | 942 | general | S | addable |
| **fused_qk_concat** | `fusions/fused_qk_concat.py` | inline | `fusions/test_fused_qk_concat.py` | bf16 | 942 | serving | M | addable |
| **fused_kv_cache** | `fusions/fused_kv_cache.py` | inline | `fusions/test_fused_kv_cache.py` | bf16/fp8 KV | 942 | serving (rope+cache) | M/L | addable |
| **kv_cache (reshape_and_cache)** | `kv_cache.py` | inline | `test_kv_cache.py` | bf16/fp8 KV | 942 | all serving | M | addable |
| **gmm (grouped matmul)** | `gmm.py` | inline | `test_gmm.py:torch_gmm/torch_tgmm` | bf16 | 942 | MoE/serving | M | addable (plain gmm; moe_fused_gemm overlaps) |
| **topk** | `topk.py` | inline | `test_topk.py` | fp | 942 | sampling/routing | S/M | addable |
| **GR ops** (jagged / norm / swiglu / hstu / addmm / position) | `generative_recommenders/ops/triton/triton_*.py` | clean/inline | (GR repo tests) | bf16/fp32 | 942 | **GR ranking** | S–L | see **§GR** (4 DONE + GR-2..4 queue) |
| **mhc (sinkhorn)** | `fusions/mhc.py` | inline | `fusions/test_mhc.py:mhc_torch` | bf16 | 942 | niche routing | L | addable (low-pri) |
| **fused_bmm_rope_kv_cache** | `fusions/fused_bmm_rope_kv_cache.py` | inline | `fusions/test_fused_bmm_rope_kv_cache.py` | bf16/fp8 | 942 | DS MLA serving | L | addable |
| **fused_reduce_qk_norm_rope_swa_write** | `fusions/fused_reduce_qk_norm_rope_swa_write.py` | inline | `fusions/test_fused_reduce_qk_norm_rope_swa_write.py` | bf16/fp8 | 942 | sliding-window serving | L | addable |
| **fused_routing_from_topk** | `fusions/fused_routing_from_topk.py` | inline | (routing helper) | int/fp | 942 | MoE | M | addable (overlaps moe routing) |
| **gather_kv_b_proj** | `gather_kv_b_proj.py` | inline | `test_gather_kv_b_proj.py:ref_gather_kv_b_proj` | FP4 layout | 950-ish | DS MLA | L | **skip** (niche, fp4 layout-heavy) |
| **fused_rearrange_sigmoid_gdr** | `gated_delta_net/fused_rearrange_sigmoid_gdr.py` | inline | `test_fused_rearrange_sigmoid_gdr.py` | bf16 | 942 | linear-attn | L | **skip** (overlaps gated_delta_rule) |

---

## §GR. generative-recommenders (commit `456ff31a76d7a96aa18f0723436897150adb8734`)

Source: `generative_recommenders/ops/triton/triton_*.py`. All gfx942 (0 gfx950-only).
DONE = 4, addable = 8, skip = 5.

### GR DONE (4)

| op | source | standalone? | upstream test | AMD dtype | arch | models | diff | status |
|---|---|---|---|---|---|---|---|---|
| **jagged_dense_bmm_broadcast_add** | `triton_jagged_dense_bmm_broadcast_add.py` | clean | (GR repo tests) | bf16/fp32 | 942 | **GR ranking** | M | **DONE** `generative_recommenders/jagged_dense_bmm_broadcast_add` (prior) |
| **jagged_dense_broadcast_add** | `triton_jagged.py` (broadcast-add) | clean | (GR repo tests) | bf16/fp32 | 942 | **GR** | S | **DONE** `generative_recommenders/jagged_dense_broadcast_add` |
| **layer_norm** | `triton_layer_norm.py` | clean | (GR repo tests) | bf16/fp32 | 942 | **GR** | S | **DONE** `generative_recommenders/layer_norm` |
| **swiglu** | `triton_swiglu.py` | clean | (GR repo tests) | bf16/fp32 | 942 | **GR** | S | **DONE** `generative_recommenders/swiglu` |

### GR addable queue (gfx942, 8)

| batch | op | source | diff | arch | status |
|---|---|---|---|---|---|
| GR-2 | **rms_norm** | `triton_rms_norm.py` | S | 942 | addable |
| GR-2 | **addmm** | `triton_addmm.py` | S/M | 942 | addable |
| GR-2 | **swish_layer_norm** | `triton_swish_layer_norm.py` | S | 942 | addable |
| GR-3 | **concat_2D_jagged** | `triton_concat_2D_jagged.py` | M | 942 | addable |
| GR-3 | **split_2D_jagged** | `triton_split_2D_jagged.py` | M | 942 | addable |
| GR-3 | **add_timestamp_positional_embeddings** | `triton_position.py` | M | 942 | addable |
| GR-4 | **hstu_attention** | `triton_hstu_attention.py` | L | 942 | addable (heavy) |
| GR-4 | **hstu_compute_output** | `triton_hstu_attention.py` | L | 942 | addable (heavy) |

### GR skips (5)

- **jagged_reduce_sum** — backward helper.
- **triton_hstu_preprocess_and_attention** — composite (wraps several kernels).
- **acc_dq** — backward.
- **swiglu TLX / TMA persistent** — SM100 (NVIDIA), wrong arch.
- **helion variants** — Helion-language, no clean FlyDSL target.

---

## §SG. SGLang (commit `9bb9d17e1ab26d50b91e5672b7ffcac800faafc7`)

Source: SGLang `python/sglang/srt/layers/.../*` Triton kernels. DONE = 21 (all gfx942).
A read-only analysis produced the prioritized **additive** candidate set below
(S1–S4 triton2flydsl; T1–T2 torch2flydsl). **Batches S1 and S2–S4 are now built**
(GDN/FLA chunk pipeline, the RadixAttention native backend, elementwise/MoE/mrope,
linear-attn/SSD, and the dsv4 FP4 quant). Two candidates remain (see the queue):
`ssd_chunk_scan` (deferred) and `mxfp8_block_scaled_matmul` (dropped — NVIDIA-only).

### SGLang DONE (21, all gfx942)

| op | dir | diff | status |
|---|---|---|---|
| **gdn_chunk_fwd_h** (Gated-DeltaNet chunk fwd, h-state) | `sglang/gdn_chunk_fwd_h` | L | **DONE** |
| **gdn_chunk_fwd_o** (Gated-DeltaNet chunk fwd, output) | `sglang/gdn_chunk_fwd_o` | L | **DONE** |
| **gdn_fused_recurrent_decode** (GDN fused recurrent decode) | `sglang/gdn_fused_recurrent_decode` | L | **DONE** |
| **gdn_l2norm_fwd** (GDN L2-norm fwd) | `sglang/gdn_l2norm_fwd` | S | **DONE** |
| **sglang_fused_moe** (SGLang fused MoE) | `sglang/sglang_fused_moe` | M | **DONE** |
| **fused_gdn_gating** (GDN input gating: g + beta_output) | `sglang/fused_gdn_gating` | S | **DONE** (S1; 9/9 shapes) |
| **chunk_local_cumsum** (scalar + vector chunk cumsum) | `sglang/chunk_local_cumsum` | M | **DONE** (S1; 10/10 shapes) |
| **chunk_scaled_dot_kkt_fwd** (gated beta·K@K^T tril) | `sglang/chunk_scaled_dot_kkt_fwd` | M | **DONE** (S1; 8/8 shapes) |
| **fused_norm_gate** (gated RMS/LayerNorm fwd) | `sglang/fused_norm_gate` | M | **DONE** (S1; 8/8 shapes) |
| **wy_fast** (recompute_w_u fwd) | `sglang/wy_fast` | M | **DONE** (S1; 8/8 shapes) |
| **merge_state** (flash-decoding state combine) | `sglang/merge_state` | S/M | **DONE** (S2; 7/7 shapes) |
| **prefill_attention** (`context_attention_fwd`) | `sglang/prefill_attention` | L | **DONE** (S2; 7/7 shapes) |
| **extend_attention** (prefill-with-KV-cache, 2-stage) | `sglang/extend_attention` | L | **DONE** (S2; 7/7, incl. MLA split-head) |
| **decode_attention** (2-stage flash-decoding) | `sglang/decode_attention` | L | **DONE** (S2; 6/6 MHA+GQA+MQA) |
| **fused_dual_residual_rmsnorm** | `sglang/fused_dual_residual_rmsnorm` | S/M | **DONE** (S3; 8/8 shapes) |
| **experts_combine** (MoE+MLP combine) | `sglang/experts_combine` | S/M | **DONE** (S3; 8/8 shapes) |
| **triton_mrope_fused** (Qwen-VL sectioned M-RoPE) | `sglang/triton_mrope_fused` | M | **DONE** (S3; 8/8, neox+gptj+interleaved) |
| **fused_moe_router** (softcapped top-k) | `sglang/fused_moe_router` | M | **DONE** (S3; 8/8, cudacore+tensorcore) |
| **lightning_attn** (MiniMax decode recurrence) | `sglang/lightning_attn` | L | **DONE** (S3; 6/6 shapes) |
| **ssd_chunk_state** (Mamba2 chunk-state fwd) | `sglang/ssd_chunk_state` | L | **DONE** (S3; 6/6 shapes) |
| **dsv4_fp4_indexer** (DeepSeek-V4 FP4 indexer quant + cache store) | `sglang/dsv4_fp4_indexer` | M | **DONE** (S4→942; 6/6 bit-exact) |

### SGLang triton2flydsl additive queue

**Batches S1 and S2–S4 — DONE** (all gfx942, all `--correctness` exit 0). S1
finishes the GDN/FLA chunk pipeline; S2–S4 add the RadixAttention native backend,
elementwise/MoE/mrope ops, linear-attn/SSD, and the dsv4 FP4 quant (see the DONE
table above).

**Remaining (2):**

| family | op | diff | arch | status |
|---|---|---|---|---|
| S3 | **ssd_chunk_scan** (Mamba2 SSD chunk scan) | L | 942 | **queued** — deferred |
| S4 | **mxfp8_block_scaled_matmul** | M | — | **dropped** — not gfx950 |

- **ssd_chunk_scan** (`mamba/ops/ssd_chunk_scan.py:_chunk_scan_fwd`): the single
  remaining gfx942-runnable kernel. It is the full SSD chunk scan combining the
  intra-chunk causal `cb`(=C·Bᵀ)-weighted accumulation, the inter-chunk
  prefix-`states` contribution via `C·exp(dA_cumsum)`, and the optional `D` skip /
  `z` gate, behind a `TRITON_22` version path. Deferred (not faked): a faithful
  standalone torch reference must reproduce all three terms from precomputed
  `cb`/`C`/`states`; it was left queued rather than ship a rushed/under-validated
  gate. The sibling `ssd_chunk_state` (the state-producing stage) **is** built.
- **mxfp8_block_scaled_matmul** (`quantization/fp8_kernel.py`): **dropped — not a
  gfx950 kernel.** It uses Triton TMA `TensorDescriptor.from_tensor` and is gated on
  `_is_sm100_supported`/`_is_sm120_supported` (NVIDIA Blackwell SM100/SM120). TMA
  tensor descriptors do not exist on CDNA4/gfx950, so the gfx950 skip-guard premise
  ("ready for a gfx950 node") is false; it cannot run on AMD at all. Removed from the
  gfx950-only set.
- **dsv4 fp4 indexer/quant** was reclassified from gfx950-only to **gfx942**: the
  indexer quantizer (`dsv4/fp4_indexer.py`) is pure integer/bitwise e2m1+ue8m0
  packing with **no** `tl.dot_scaled`/scaled-MFMA, so it compiles and runs on gfx942
  and was built + validated bit-exactly there (`sglang/dsv4_fp4_indexer`).

### SGLang torch2flydsl candidates (pure-torch `Model` source — note only)

These belong to the `torch2flydsl` suite, not counted in the triton2flydsl totals:

- **T1:** `rms_norm_gated` (copy `fla/layernorm_gated.py:rms_norm_ref`),
  DeepSeek grouped/biased top-k routing (`moe/topk.py`),
  `fused_dual_residual_rmsnorm`, sectioned `mrope`.
- **T2 (L):** GDN recurrence ref, Mamba2 SSD ref, lightning-attn ref.

### SGLang additive-vs-duplicate verdict

- **Additive value (build these):** architecture families aiter lacks —
  Gated-DeltaNet / linear-attn, Mamba2 SSD, MiniMax lightning-attn, the SGLang
  native RadixAttention Triton backend, gated / dual-residual RMSNorm, DeepSeek
  grouped-topk routing, and `mrope`.
- **SKIP (duplicates aiter):** basic GEMM / quant / silu-gelu / norms / softmax,
  `aiter_unified_attention`, `w8a8_block_fp8_matmul`, and `fused_moe` (already
  done). **Also skip** EP / distributed MoE, LoRA, KV-cache metadata,
  speculative-decode index kernels, constrained / grammar, diffusion,
  gluon / tilelang / SM100, and all backward kernels.

---

## Prioritized build queue (gfx942 first)

Each batch front-loads high-frequency ops every target model runs, reuses an
established harness skeleton, and has a clean upstream torch ref to mirror.

### Batch 2 — bread-and-butter (all gfx942, clean torch refs) — DONE
1. `rope_fwd` (neox/gptj, sbhd fwd) — `rope/test_rope.py:ref_rope_sbhd_fwd` — every model. **DONE** (`aiter/rope_fwd`, 48/48 cfgs).
2. `gemm_a8w8_blockscale` (fp8 128×128) — `test_gemm_a8w8_blockscale.py:run_torch` — **DeepSeek-V3**. **DONE** (`aiter/gemm_a8w8_blockscale`, 7/7 shapes, e4m3fnuz).
3. `gemm_a8w8` (fp8 per-token) — `test_gemm_a8w8.py:run_torch` — Llama/Qwen fp8. **DONE** (`aiter/gemm_a8w8`, 18/18 shape×bias, e4m3fnuz).
4. `fused_silu_mul` (silu·mul) — `fusions/test_fused_silu_mul.py` — all MoE/FFN. **DONE** (`aiter/fused_silu_mul`, 18/18 shape×dtype).
5. `fused_clamp_act_mul` (clamped swiglu) — `fusions/test_fused_clamp_act_mul.py` — **GPT-OSS**. **DONE** (`aiter/fused_clamp_act_mul`, non-quant path, 60/60 cfgs).

### Batch 3 — quant + batched/FFN GEMM (gfx942)
- `batched_gemm_bf16`, `batched_gemm_a8w8` — `gemm/batched/test_*`. **DONE** (sub-batch 1).
- `ff_a16w16` (ungated) — `gemm/feed_forward/test_ff_a16w16*.py`. **DONE** (sub-batch 1).
- `gemm_afp8wfp8` — `gemm/basic/test_*`. **DONE** (sub-batch 1, gfx950-only skip-guard — uses `tl.dot_scaled`).
  - `dynamic_mxfp8_quant`, `per_token/per_tensor fp8 quant` — `quant/test_quant*.py`. **DONE** (sub-batch 2).
- `gemm_a16w8_blockscale` — `gemm/basic/test_*` (uses `tl.dot`, gfx942). **DONE** (sub-batch 2).
- `ff_a16w16_fused_gated`/`_fused_ungated` (atomic-add fused single-kernel variants). still addable.

**Batch 3 complete.** Next up: Batch 4 (attention/serving) then Batch 5 (stateful/rec).

### Batch 4 — attention + serving (gfx942, L)
- `mha_with_sink` (**GPT-OSS** attention sink), `extend_attention`, `prefill_attention`,
  `pa_decode` (+ `pa_prefill`), `kv_cache (reshape_and_cache)`.

### Batch 5 — stateful / rec / fused long-tail (gfx942, M/L)
- `causal_conv1d` (Mamba), `gated_delta_rule` (**Qwen3-Next**), `gmm`,
  `fused_qk_concat`, `fused_kv_cache`, GR `hstu_attention` + GR jagged/addmm/swiglu.

### Batch 6 — gfx950-only (FP4 / MX scaled-dot — skip-guard on gfx942)
- `gemm_afp4wfp4`, `gemm_a16wfp4`, `gemm_a8wfp4` (+ batched + fused variants),
  `moe_op_mxfp4(_silu_fused)`, `moe_op_gemm_a4w4`, `dynamic_mxfp4/nvfp4 quant`,
  `fused_mxfp4_quant`, `act_mul_and_mxfp4_quant`. All tagged
  `supported_archs: [gfx950]` + harness arch-guard (`SKIPPED`/exit 0 on gfx942),
  exactly like the existing `fav3_sage_mxfp4`.

---

## Skip list (poor fit — do NOT build)

- **Backward / training**: `mha_fused_bwd`, `mha_onekernel_bwd`, any `*_bwd`
  (RMSNorm/LayerNorm/RoPE backward kernels) — this is an inference kit.
- **Gluon-language**: everything under `_gluon_kernels/`, `gluon/`, the gfx1250
  gluon GEMM backends, `op_tests/.../test_pa_decode_gluon.py` — Triton-Gluon
  dependency, no clean FlyDSL target.
- **gfx1250-only**: gluon a16w16, gfx1250 grouped MoE, nvfp4 — wrong arch for the
  gfx942/gfx950 target.
- **Multi-GPU / distributed**: `comms/`, `custom_all_reduce`, `quick_all_reduce`,
  MoE EP/DP-share.
- **Infra / helpers, not ops**: `moe/reduce.py`, `moe/moe_routing/bitmatrix.py`,
  `utils/*`, tuning/config loaders, `triton_metadata_redirect`.
- **Niche / layout-heavy / overlapping**: `gather_kv_b_proj` (fp4 gather),
  `fused_rearrange_sigmoid_gdr` (overlaps `gated_delta_rule`),
  `sage_attention_quant_wrappers` (used by the DONE `fav3_sage`).

---

## Notes & caveats

- **fp8 dtype is arch-specific.** gfx942 fp8 = `e4m3fnuz` (max ≈ 240); gfx950 =
  `e4m3fn` (max = 448). Any fp8 task harness must build its torch reference with
  the arch-matched dtype (see HANDOFF §D action items) and keep the tight gate.
- **FP4 / MX scaled-dot is gfx950-only.** Kernels using `tl.dot_scaled`
  (e2m1/e8m0 microscale) fail to compile on gfx942 ("Unsupported DotScaleOp").
  Tag those `supported_archs: [gfx950]` with the harness arch-guard.
- **Standalone extraction pattern** (proven on the 5 built this batch): copy the
  device kernel verbatim from `_triton_kernels/...`, inline the small util
  helpers (`make_kernel_repr`/`_sanitize_constexpr_value`, `pid_grid`/`remap_xcd`,
  `get_num_sms`→`multi_processor_count`, `get_num_xcds`→8), drop the
  quant/fused/backward branches not under test, and replace any on-disk tuned
  config lookup with a static config (force `NUM_KSPLIT == 1` for GEMMs to avoid
  the split-K reduce pass). The `moe_fused_gemm` task is the reference for the
  inlined pid/XCD helpers.
- **Harness correctness for these ops is a real gate, not a no-op:** it runs the
  Triton kernel on real shapes and asserts finiteness AND closeness to a trivial
  inline torch reference at the upstream tolerance (1e-2 bf16/fp16 for
  norm/softmax; atol=1e-1/rtol=1e-2 bf16 for GEMM). No separate torch reference
  *file* is shipped (the suite contract), and the gate is never widened.
