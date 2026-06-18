# AITER Op Surface Catalog — for `torch2flydsl` task generation

Inventory of the AITER PyTorch-facing op surface to drive batch creation of
`torch2flydsl` tasks (PyTorch reference `model.py` in KernelBench format →
FlyDSL target `kernel.py`). References are intended to be **AMD-runtime-faithful
(option b)**: the `model.py` reference should reproduce the real
quant/dtype/layout semantics the op runs with on AMD via aiter, not an idealized
fp32 version.

- **AITER source**: `/workspaces/WS/aiter` (HEAD `e773193041`).
- **Existing tasks**: `/workspaces/WS/AgentKernelArena/tasks/torch2flydsl/` (48 dirs)
  + `tasks/triton2flydsl/` (10 dirs) on branch `torch2flydsl`.
- **Status is now LIVE** — most cataloged ops have been built. See the
  *Built-task index* below for the authoritative dir-by-dir mapping; the
  per-row `status` columns mirror it. The *not-yet* / *skip* rows remain the
  accurate remaining-work / poor-fit list.

---

## Legend

| Field | Meaning |
|---|---|
| **op / API** | Op name + the aiter PyTorch entry point (`module:function`). Paths are relative to `aiter/aiter/ops/` unless noted. |
| **torch ref** | Location of the (b)-faithful PyTorch reference used in the op's test (`op_tests/<file>:<fn>`). `—` = no standalone torch ref (only a backend-vs-backend check). |
| **AMD semantics** | dtypes / quantization / layout the op actually runs on AMD (what the reference must emulate). |
| **shapes / config** | Real shape source: `configs/*.csv` or `configs/model_configs/*.csv`, else op_test enum. |
| **models** | Best-effort: which production models exercise it (DS=DeepSeek-V3/V4, GPT-OSS, Llama, Qwen, GLM, Kimi). |
| **status** | `DONE` (+ task) / `not-yet` / `skip` (poor fit). |
| **difficulty** | Port difficulty + key gotchas. |

**Difficulty scale**: `S` simple (single GEMM/elementwise, fp ref) · `M` medium
(quant emulation or fused multi-step) · `L` large (stateful / multi-kernel /
metadata) · `XL` impractical (gluon-only, gfx1250-only, no torch ref).

---

## Summary counts (as built on branch `torch2flydsl`)

DONE counts below include both suites: `torch2flydsl` (PyTorch `Model` source)
and `triton2flydsl` (standalone Triton source; FlyDSL is GEAK's target, added
later). 58 task dirs total (48 + 10).

| Category | Distinct ops cataloged | DONE | not-yet | skip/poor-fit |
|---|--:|--:|--:|--:|
| GEMM | 12 | 9 | 1 | 2 |
| MoE (gemm + routing/sort/topk) | 11 | 9 | 1 | 1 |
| Attention (MHA/MLA/paged/sage/logits) | 13 | 6 (triton2flydsl) | 4 | 3 |
| Norm (rms/layer/group) | 7 | 5 | 2 | 0 |
| RoPE / fused qk-norm-rope | 9 | 3 | 5 | 1 |
| Quant | 9 | 6 | 2 | 1 |
| Activation | 6 | 5 | 1 | 0 |
| KV-cache | 7 | 0 | 4 | 3 |
| Sampling / topk | 6 | 0 | 4 | 2 |
| Other (conv / linear-attn / mhc / gmm / misc) | 8 | 2 | 3 | 3 |
| **Total** | **~88** | **~45** | **27** | **~16** |

---

## Built-task index (authoritative dir → op map)

**`tasks/torch2flydsl/` — 48 dirs.** 8 ship an inlined FlyDSL `kernel.py`
(apple-to-apple validated, §6 YES); 40 are model-only (FlyDSL is GEAK's target).

| Inlined `kernel.py` (8) | Model-only — GEAK target (40) |
|---|---|
| `hgemm_kernel`, `gemm_a8w8_bpreshuffle_kernel`, `moe_kernel`, `moe_a8w4_kernel`, `moe_swiglu_kernel`, `moe_sorting_kernel`, `qk_norm_rope_quant_kernel`, `jagged_dense_bmm_kernel` | GEMM: `gemm_a8w8`, `gemm_a8w8_blockscale`, `gemm_a4w4`, `gemm_a16w8_blockscale`, `gemm_a8w8_per_token_scale`, `gemm_a16wfp4`, `gemm_a8wfp4`, `gemm_afp4wfp4`, `gemm_afp8wfp8`, `batched_gemm_bf16`, `batched_gemm_a8w8` · MoE: `fmoe_fp8_blockscale_g1u1`, `fmoe_g1u1_tkw1`, `moe_2stage_generic`, `moe_topk_softmax`, `moe_topk_sigmoid`, `moe_topk_softplus`, `moe_biased_grouped_topk` · Norm: `rmsnorm2d`, `fused_add_rmsnorm`, `rmsnorm2d_dynamicquant`, `rmsnorm2d_smoothquant`, `layernorm2d`, `layernorm2d_with_add` · RoPE: `rope_fwd`, `rope_2d_fwd`, `rope_thd_fwd` · Quant: `quant_mxfp4`, `dynamic_mxfp8_quant`, `per_token_fp8_quant`, `per_tensor_fp8_quant`, `per_1x128_fp8_quant`, `per_token_i8_quant`, `smoothquant` · Activation: `silu_and_mul`, `swiglu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`, `gelu_fast`, `silu_and_mul_quant` |

**`tasks/triton2flydsl/` — 10 dirs** (each: `<name>.py` Triton/torch source +
`test_kernel_harness.py` + `config.yaml`; no torch reference, no `kernel.py`):
`aiter/mha`, `aiter/mla`, `aiter/fav3_sage`, `aiter/fav3_sage_mxfp4`,
`aiter/fp8_mqa_logits`, `aiter/unified_attention`,
`aiter/unified_attention_sparse_mla`, `aiter/moe_fused_gemm`,
`aiter/moe_routing_sigmoid_top1`,
`generative_recommenders/jagged_dense_bmm_broadcast_add`.

**gfx950-only (FP4 — CDNA3/gfx942 has no native FP4 MFMA):**
`gemm_a4w4`, `moe_kernel`, `moe_swiglu_kernel`, `moe_a8w4_kernel` (fp4 weight),
`gemm_a16wfp4`, `gemm_a8wfp4`, `gemm_afp4wfp4`, `quant_mxfp4`, and triton
`fav3_sage_mxfp4` (9 total). See `HANDOFF_GFX942.md` §D.

---

## 1. GEMM

The big real-shape source is `configs/model_configs/` (dsv3/dsv4, qwen3*, glm*,
gptoss, kimi). Each dtype variant has a `*_untuned_gemm*` (shapes) + `*_tuned_gemm*`
(shapes + best config) pair.

| op / API | torch ref | AMD semantics | shapes / config | models | status | difficulty |
|---|---|---|---|---|---|---|
| **bf16 GEMM** `gemm_op_a16w16:gemm_a16w16_asm` (also `opus/gemm_op_a16w16`) | `test_gemm_a16w16.py:run_torch` / `test_opus_a16w16_gemm.py:_torch_ref` | a16w16 bf16/fp16, fp32 accum, optional bias/scale | `*_bf16_*_gemm*.csv` (dsv3, dsv4, glm47, glm5, gptoss, kimi) | all dense models | **DONE** `hgemm_kernel` | S |
| **a8w8 per-token/tensor fp8** `gemm_op_a8w8:gemm_a8w8` | `test_gemm_a8w8.py:run_torch` | fp8 e4m3 act+weight, per-token/per-tensor scale, fp32 accum→bf16 | `a8w8_*tuned_gemm.csv` | Llama/Qwen fp8 | **DONE** `gemm_a8w8_kernel` (model-only) | M (fp8 quant emul) |
| **a8w8 blockscale fp8** `gemm_op_a8w8:gemm_a8w8_blockscale` | `test_gemm_a8w8_blockscale.py:run_torch`/`run_torch2`; cktile variant `..._cktile_aq_rowmajor.py:torch_reference` | fp8 e4m3, 128×128 block scales | `a8w8_blockscale_*tuned_gemm_*.csv` (ds_v3, qwen3_235b, qwen3_5_397b, qwen3_next, qwen3_vl, qwen36) | **DeepSeek-V3**, Qwen3 family | **DONE** `gemm_a8w8_blockscale_kernel` (model-only) | M |
| **a8w8 bpreshuffle fp8** `gemm_op_a8w8:gemm_a8w8_bpreshuffle` | `test_gemm_a8w8_bpreshuffle_pad_k.py` (backend ref) | fp8 + weight preshuffle | `a8w8_bpreshuffle_*tuned_gemm*.csv`, `dsv3_a8w8_bpreshuffle_*`, `glm47` | DS-V3, GLM-4.7 | **DONE** `gemm_a8w8_bpreshuffle_kernel` (inlined `kernel.py`) | M (preshuffle layout) |
| **a8w8 blockscale bpreshuffle** `gemm_op_a8w8:gemm_a8w8_blockscale_bpreshuffle` | shares `test_gemm_a8w8_blockscale.py` | fp8 blockscale + preshuffle | `a8w8_blockscale_bpreshuffle_*tuned_gemm_{dsv3,qwen3_235b,qwen3.5_397b}.csv`, `dsv4_*`, `glm5_*` | DS-V3/V4, Qwen3, GLM-5 | not-yet | M |
| **a4w4 mxfp4 GEMM** `gemm_op_a4w4:gemm_a4w4` / `gemm_a4w4_blockscale` | `test_gemm_a4w4.py:run_torch` | mxfp4 (e2m1) act+weight, e8m0 block scale, weight preshuffle | `a4w4_blockscale_*tuned_gemm.csv`, `dsv3_a4w4_blockscale_*` | DeepSeek-V3 fp4 | **DONE** `gemm_a4w4_kernel` (model-only; **gfx950-only**) | M (reuse mxfp4 emul from `moe_kernel`) |
| **batched a8w8 fp8** `batched_gemm_op_a8w8` | `test_batched_gemm_a8w8.py:run_torch` | batched fp8 a8w8 | `a8w8_*tuned_batched_gemm.csv` | attention-proj batched | **DONE** `batched_gemm_a8w8_kernel` (model-only) | M |
| **batched bf16** `batched_gemm_op_bf16` | `test_batched_gemm_bf16.py:run_torch` | batched bf16 | `bf16_*tuned_batched_gemm.csv` | generic | **DONE** `batched_gemm_bf16_kernel` (model-only) | S |
| **deepgemm (fp8 blockscale)** `deepgemm:*` | `test_deepgemm.py:run_torch` | fp8 e4m3 blockwise (DeepGEMM-style) | shares a8w8_blockscale csvs | DeepSeek | not-yet | M |
| **a8w8 ASM/flatmm variants** `gemm_op_a8w8:{gemm_a8w8_asm, flatmm_a8w8_blockscale_asm}` | same a8w8 tests | fp8, ASM kernels | `asm_a8w8_gemm.csv` | — | not-yet (low priority; dup of a8w8) | M |
| **gemm codegen / tune harness** `gemm_op_*:*_tune` | `test_gemm_codegen.py`, `test_pretune.py` | tuning infra, not an op | — | — | **skip** (infra) | XL |
| **gradlib tuned GEMM** `gradlib` | — | tuning library | — | — | **skip** (infra) | XL |

> **Long tail (triton GEMM dtype variants)** in `ops/triton/gemm/basic/` —
> all **DONE** as model-only torch2flydsl tasks: `gemm_afp8wfp8_kernel`,
> `gemm_a8wfp4_kernel`, `gemm_a16wfp4_kernel`, `gemm_afp4wfp4_kernel`,
> `gemm_a16w8_blockscale_kernel`, `gemm_a8w8_per_token_scale_kernel`. Each
> references `op_tests/triton_tests/gemm/...` (`run_torch`/`run_torch_emulation`).
> The FP4-weight variants (`a16wfp4`, `a8wfp4`, `afp4wfp4`) are **gfx950-only**.

---

## 2. MoE (fused MoE GEMMs + routing / sorting / topk)

Real shapes: `configs/*fmoe*.csv` and `model_configs/*_fmoe_*.csv`
(dsv3_fp4, dsv4_fp8fp4, gptoss_fp4/fp8fp4, glm47/glm5/minimax/qwen3 fp8/fp8-blockscale, kimik2_fp4).

| op / API | torch ref | AMD semantics | shapes / config | models | status | difficulty |
|---|---|---|---|---|---|---|
| **fused MoE a4w4 (mxfp4)** `moe_op:ck_moe_stage1/2`, `moe_op_gemm_a4w4` | `op_tests/test_moe.py` (`torch_moe`), `test_flydsl_grouped_gemm_gfx1250.py:_torch_moe_ref` | mxfp4 act+weight, e8m0, 2-stage grouped GEMM, silu-gate | `dsv3_fp4_*fmoe.csv`, `kimik2_fp4_*` | DS-V3, Kimi-K2 | **DONE** `moe_kernel` | M |
| **fused MoE a8w4** `moe_op_gemm_a8w4` (triton) / mixed `moe_cktile2stages` | `test_moe.py`, `triton_tests/moe/test_moe_gemm_a8w4.py` | fp8 act · fp4 weight | `dsv4_fp8fp4_*`, `gptoss_fp8fp4_*` | DS-V4, GPT-OSS | **DONE** `moe_a8w4_kernel` | M |
| **fused MoE swiglu (fp4)** mixed moe + swiglu act | `test_moe.py` (swiglu path) | fp4, swiglu (clamped) activation | `gptoss_fp4_*fmoe.csv` | GPT-OSS | **DONE** `moe_swiglu_kernel` | M |
| **fused MoE a8w8 blockscale** `moe_op:fmoe_fp8_blockscale_g1u1`, `moe_op_gemm_a8w8_blockscale` | `test_moe_blockscale.py` | fp8 blockscale, g1u1 | `a8w8_blockscale_*fmoe_{ds_v3,glm5,minimax-m2_5,qwen3_235b,qwen3_5_397b}.csv` | DS-V3, GLM-5, Qwen3, MiniMax | **DONE** `fmoe_fp8_blockscale_g1u1_kernel` (model-only) | M |
| **fused MoE 2-stage (generic ck)** `moe_op:moe_stage1_g1u1`, `ck_moe_stage1/2_fwd` | `test_moe_2stage.py` | bf16/fp8/fp4 grouped 2-stage | `*fmoe*.csv` | all MoE models | **DONE** `moe_2stage_generic_kernel` (model-only) | M |
| **MoE tkw1 (token-weight stage1)** `moe_op:fmoe_g1u1_tkw1` | `test_moe_tkw1.py` | per-token weighting at stage1 | fmoe csvs | DS variants | **DONE** `fmoe_g1u1_tkw1_kernel` (model-only; **TOL=3.5e-2 exception**, see HANDOFF §E) | M |
| **MoE sorting** `moe_sorting:moe_sorting_fwd` | `test_moe_sorting.py`, `test_moe_sorting_mxfp4.py:run_torch` | token→expert sort/dispatch, unit blocks | n/a (counts) | all MoE | **DONE** `moe_sorting_kernel` | M |
| **MoE topk gating (softmax)** `topk.py:topk_gating`, `moe_op:topk_softmax` | `test_moe_topk_gating.py:run_torch_softmax`, `test_moeTopkSoftmax.py` | softmax router + top-k renorm | n/a | Llama/Qwen/GPT-OSS | **DONE** `moe_topk_softmax_kernel` (model-only) | S |
| **MoE biased grouped topk** `topk.py:biased_grouped_topk` (+ `_torch`) | `topk.py:biased_grouped_topk_torch` (ref ships in module), `test_topk_*` | grouped/biased top-k (DS routing), sigmoid scoring | n/a | **DeepSeek-V3** routing | **DONE** `moe_biased_grouped_topk_kernel` (model-only) | M |
| **MoE topk softplus / sigmoid-top1** `topk.py:topk_softplus`; triton `moe_routing_sigmoid_top1` | `test_moe_topk_gating.py:run_torch_softplus`, `torch_compile/test_compile_moe_routing.py:torch_routing_sigmoid_top1` | softplus / sigmoid-top1 routing | n/a | GPT-OSS-ish | **DONE** `moe_topk_softplus_kernel`, `moe_topk_sigmoid_kernel` (model-only); sigmoid-top1 also `triton2flydsl/aiter/moe_routing_sigmoid_top1` | S |
| **MoE EP / DP-share / local-expert** `moe_op` variants | `test_moe_ep.py:torch_moe_test`, `test_moe_dp_share_expert.py`, `test_moe_local_expert_ids.py` | multi-GPU expert parallel | n/a | distributed serving | **skip** (multi-GPU/distributed) | XL |

---

## 3. Attention (MHA / MLA / paged / sage / logits)

> **Attention is being built in the `triton2flydsl` suite** (standalone Triton
> source from aiter/Meta; triton+torch only, faithful to upstream, no torch
> reference; FlyDSL is GEAK's target added later). 6 dirs DONE:
> `aiter/mha`, `aiter/mla`, `aiter/unified_attention`,
> `aiter/unified_attention_sparse_mla`, `aiter/fp8_mqa_logits`,
> `aiter/fav3_sage` (+`fav3_sage_mxfp4`, gfx950-only). Torch2flydsl
> (PyTorch-`Model`-as-source) attention references are the *not-yet* rows below.

| op / API | torch ref | AMD semantics | shapes / config | models | status | difficulty |
|---|---|---|---|---|---|---|
| **flash MHA fwd (bf16)** `mha.py:flash_attn_func` / `mha_fwd` | `test_mha.py:run_torch` | bf16 causal/non-causal SDPA, GQA | op_test enum (b,h,s,d) | all dense | **DONE (triton2flydsl)** `aiter/mha`; torch2flydsl ref not-yet | M (softmax+causal) |
| **flash MHA varlen** `mha.py:flash_attn_varlen_func`/`mha_varlen_fwd` | `test_mha_varlen.py`, `test_mha_flydsl_varlen.py:_ref_mha_varlen` | varlen (cu_seqlens), bf16 | op_test enum | serving | not-yet | M (varlen indexing) |
| **flash MHA fp8** `mha_v3.py:flash_attn_fp8_func` | `test_mha_fp8.py`, `test_mha_varlen_fp8.py` | fp8 e4m3 q/k/v, per-tensor scale | op_test enum | fp8 inference | not-yet | M |
| **MHA fwd with attention sink** `mha.py:fmha_fwd_with_sink_asm(_varlen)` | `test_fmha_fwd_with_sink_asm.py:_ref_attn`, `..._varlen_asm.py:_ref_varlen` | bf16 + attention sink (GPT-OSS) | op_test enum | **GPT-OSS** | not-yet | M |
| **MLA decode/prefill** `attention.py:{mla_decode_stage1_asm_fwd, mla_prefill_asm_fwd, mla_reduce_v1}`; triton `mla.py` | `test_mla.py:torch_mla_extend`, `test_mla_prefill_ps.py`, `test_mla_persistent.py` | MLA (compressed KV, rope split), bf16/fp8 KV, split-KV + reduce | op_test enum | **DeepSeek-V3/V4** | **DONE (triton2flydsl)** `aiter/mla` | L (metadata + split-kv reduce) |
| **MLA sparse** `test_mla_sparse.py` + `unified_attention_sparse_mla` | `test_mla_sparse.py` | sparse top-k KV select | op_test enum | DS sparse | **DONE (triton2flydsl)** `aiter/unified_attention_sparse_mla` | L |
| **paged attention (decode)** `attention.py:{paged_attention_v1, paged_attention_ragged, pa_fwd_asm, pa_decode_bf16_asm}` | `test_pa.py`, `test_pa_ragged.py:run_torch`, `test_pa_decode_bf16_asm.py:ref_pa_decode` | paged KV-cache, bf16/fp8 KV, block tables | op_test enum | all serving | not-yet | L (paged KV + block tables) |
| **fp8 paged MQA logits (DeepGEMM)** `pa_mqa_logits.py:deepgemm_fp8_paged_mqa_logits`; triton `fp8_mqa_logits` | `triton_tests/attention/test_fp8_mqa_logits.py:ref_fp8_mqa_logits`, `bench_deepgemm_attention.py:ref_fp8_paged_mqa_logits` | fp8 MQA logit scoring for sparse/indexer | op_test enum | **DeepSeek** indexer | **DONE (triton2flydsl)** `aiter/fp8_mqa_logits` | L |
| **sage attention (mxfp4)** triton `fav3_sage*`, `sage_attention_quant_wrappers` | `test_fav3_sage.py` (`test_sage_*_vs_reference`) | sage quant (mxfp4) block-sparse attention | op_test enum | long-context | **DONE (triton2flydsl)** `aiter/fav3_sage` (bf16) + `aiter/fav3_sage_mxfp4` (gfx950-only); upstream `_sage_fwd_mask` omits `USE_BIAS` — see HANDOFF §E | XL |
| **unified attention** triton `unified_attention.py` | `triton_tests/attention/test_unified_attention.py:ref_paged_attn` | unified prefill+decode paged | op_test enum | vLLM-style | **DONE (triton2flydsl)** `aiter/unified_attention` | L |
| **MHA bwd / fmha_v3 bwd** `mha.py:{mha_bwd, fmha_v3_bwd, ...}` | `test_mha.py` bwd path | training backward | — | training | **skip** (bwd, training-only) | XL |
| **pa decode (gluon/sparse)** triton `pa_decode*`, gluon `pa_mqa_logits` | `test_pa_decode_gluon.py`, `test_pa_decode_sparse.py` | gluon-lang kernels | — | — | **skip** (gluon dependency) | XL |
| **batch prefill** `mha.py:cmdGenFunc_mha_batch_prefill` | `test_batch_prefill.py:ref_masked_attention` | varlen batched prefill | op_test enum | serving | not-yet (overlaps varlen MHA) | L |

---

## 4. Normalization (RMSNorm / LayerNorm / GroupNorm)

| op / API | torch ref | AMD semantics | models | status | difficulty |
|---|---|---|---|---|---|
| **RMSNorm 2D** `rmsnorm.py:rms_norm / rmsnorm2d_fwd` | `test_rmsnorm2d.py` (`run_torch`); triton `torch_rmsnorm` | bf16 in/out, fp32 reduce | all transformers | **DONE** `rmsnorm2d_kernel` (model-only) | **S** |
| **Fused add + RMSNorm** `rmsnorm.py:rmsnorm2d_fwd_with_add` / `add_rmsnorm` | `test_rmsnorm2dFusedAddQuant.py`; triton `test_fused_rmsnorm_add.py:run_torch` | residual-add + rmsnorm fused | all transformers | **DONE** `fused_add_rmsnorm_kernel` (model-only) | **S** |
| **RMSNorm + dynamic quant (fp8)** `rmsnorm.py:rmsnorm2d_fwd_with_dynamicquant` | `test_rmsnorm2dFusedAddQuant.py` | rmsnorm → fp8 per-token quant | fp8 inference | **DONE** `rmsnorm2d_dynamicquant_kernel` (model-only) | M |
| **RMSNorm + smoothquant** `rmsnorm.py:rmsnorm2d_fwd_with_smoothquant` | `test_smoothquant.py` | rmsnorm → smoothquant int8/fp8 | quantized | **DONE** `rmsnorm2d_smoothquant_kernel` (model-only) | M |
| **LayerNorm 2D (+add/+smoothquant)** `norm.py:{layernorm2d_fwd, _with_add, _with_smoothquant}` | `test_layernorm2d.py`, `test_layernorm2dFusedAddQuant.py` | bf16 layernorm, optional fused add + quant | GPT-style | **DONE** `layernorm2d_kernel`, `layernorm2d_with_add_kernel` (model-only) | **S**/M |
| **GroupNorm** `groupnorm.py` | `test_groupnorm.py` | group norm | vision/audio | not-yet | S |
| **Gated RMSNorm + fp8 group quant** `gated_rmsnorm_fp8_group_quant.py` | `test_gated_rmsnorm_fp8_group_quant.py`; triton `fused_rms_gated_fp8_group_quant.py:ref_rmsnorm_quant` | gated rmsnorm → fp8 group quant | DS/Qwen fp8 | not-yet | M |

---

## 5. RoPE / fused QK-norm-RoPE

| op / API | torch ref | AMD semantics | models | status | difficulty |
|---|---|---|---|---|---|
| **RoPE fwd (neox/gptj)** `rope.py:rope_fwd / rope_cached_fwd` | `test_rope.py`; triton `torch_rope_neox/gptj` | bf16 rotary, cached cos/sin, thd/2d variants | all | **DONE** `rope_fwd_kernel`, `rope_2d_fwd_kernel`, `rope_thd_fwd_kernel` (model-only) | **S** |
| **fused QK-norm + RoPE + quant** `fused_qk_norm_rope_cache_quant.py` | `test_fused_qk_norm_rope_cache_quant.py:run_torch_qk_norm_rope_cache_quant_shuffle` (+1way/2way refs) | rmsnorm(q,k) + rope + fp8 KV quant + cache shuffle | Qwen/DS | **DONE** `qk_norm_rope_quant_kernel` | M |
| **fused QK-norm + mRoPE + cache quant** `fused_qk_norm_mrope_cache_quant.py` | `test_fused_qk_norm_mrope_cache_quant.py:run_torch_mrope_3d_rms_set_kv_shuffle` | 3D mRoPE (multimodal) + fp8 cache | **Qwen-VL** | not-yet | M |
| **fused QK rmsnorm + group quant** `fused_qk_rmsnorm_group_quant.py` | `test_fused_qk_rmsnorm_group_quant.py:run_torch_ref` (+ fp8/fp4 per-group refs) | qk rmsnorm → fp8/fp4 per-group quant | DS/Qwen | not-yet | M |
| **fused QK-norm idx-rqknorm** `fused_qknorm_idxrqknorm.py` | `test_fused_qknorm_idxrqknorm.py:norm_rope_ref` | indexer qk-norm + rope (DS indexer) | DeepSeek indexer | not-yet | M |
| **rotate + fp4 quant (+rope)** `quant.py:rope_rotate_activation_fp4quant_inplace` | `test_rotate_fp4quant.py:rope_rotate_fp4quant_inplace_torch` | hadamard rotate → fp4 quant, optional rope | fp4 inference | not-yet | M |
| **fused QK-norm-rope 2way per-head** `test_fused_qk_norm_rope_2way_perhead.py` | `:_torch_per_head_fp8_quant` | per-head fp8 quant + 2-way rope | — | not-yet | M |
| **fused qkv split + qk rope (+cache)** triton `rope/fused_qkv_split_qk_rope.py`, `..._norm_rope_cache.py` | `triton_tests/rope/test_fused_qkv_split_qk_rope.py:run_torch` | split fused QKV, rope on q/k, cache | serving | not-yet | M |
| **rope bwd** `rope.py:rope_bwd*` | `test_rope.py` bwd | training | — | **skip** (bwd) | XL |

---

## 6. Quant (standalone quantizers)

| op / API | torch ref | AMD semantics | models | status | difficulty |
|---|---|---|---|---|---|
| **mxfp4 dynamic quant** `quant.py:quant_mxfp4 / per_1x32_f4_quant`; triton `dynamic_mxfp4_quant` | `test_quant_mxfp4.py:ref_quant_mxfp4(_even_round)`; triton `torch_dynamic_mxfp4_quant` | f32→e2m1 + e8m0 per-1x32 block scale | DS/GPT-OSS fp4 | **DONE** `quant_mxfp4_kernel` (model-only; **gfx950-only**; GEMM uses EVEN-mode e8m0) | **S** (emul exists in `moe_kernel`) |
| **mxfp8 dynamic quant** triton `quant.py:dynamic_mxfp8_quant` | `triton_tests/quant/test_quant_mxfp8.py:torch_mxfp8_quant_from_fp32` | f32→e4m3 + e8m0 block scale | fp8 | **DONE** `dynamic_mxfp8_quant_kernel` (model-only) | S |
| **nvfp4 dynamic quant** triton `dynamic_nvfp4_quant` | `triton_tests/quant/test_quant_mxfp4.py:torch_dequant_nvfp4` | nvfp4 (e2m1 + fp8 scale) | NV-fmt interop | not-yet | M |
| **fp8 per-token / per-tensor quant** `quant.py:{dynamic_per_token_scaled_quant, per_tensor_quant}` | `test_quant.py`; triton `torch_dynamic_per_token_quant_fp8` | fp8 e4m3 per-token/tensor scale (+ int8 per-token variant) | all fp8 | **DONE** `per_token_fp8_quant_kernel`, `per_tensor_fp8_quant_kernel`, `per_token_i8_quant_kernel` (model-only) | **S** |
| **fp8 per-group quant** `quant.py:dynamic_per_group_scaled_quant(_fp4)` | `test_quant.py` | per-group (128) fp8/fp4 scale | DS blockscale | **DONE** `per_1x128_fp8_quant_kernel` (model-only) | M |
| **smoothquant** `quant.py:smoothquant_fwd / moe_smoothquant_fwd` | `test_smoothquant.py` | int8 smoothquant (act·scale), truncate-toward-zero | quantized serving | **DONE** `smoothquant_kernel` (model-only) | M |
| **fused RMS → mxfp4/mxfp8 quant** triton `fused_mxfp4_quant.py:fused_rms_mxfp4_quant`, `fused_mxfp8_quant.py` | `triton_tests/quant/test_quant_mxfp8.py:torch_rmsnorm_mxfp8_quant`; `fused_clamp_act_mul` ref | rmsnorm + block quant fused | fp4/fp8 inference | not-yet | M |
| **fused reduce/act-mul → fp8/fp4 quant** triton `fused_fp8_quant.py`, `fused_mxfp4_quant.py` | `triton_tests/quant/test_fused_fp8_quant.py:run_torch_*` | act-mul/reduce fused with group quant | MoE pre-quant | not-yet | M |
| **fp8_legacy_to_mxfp8 / requant** triton `fp8_legacy_to_mxfp8` | `test_quant_mxfp8.py:torch_fp8_legacy_to_mxfp8` | fnuz→mxfp8 re-encode | format interop | **skip** (niche interop) | M |

---

## 7. Activation (fused `*_and_mul`)

All are simple elementwise; `model.py` refs are trivial (the torch refs already
ship in `test_activation.py`). Good **first-batch warmups**.

| op / API | torch ref | AMD semantics | status | difficulty |
|---|---|---|---|---|
| **silu_and_mul** `activation.py:silu_and_mul` | `test_activation.py:torch_silu_and_mul` (supports `limit` clamp = GPT-OSS swiglu) | bf16, gate·up split | **DONE** `silu_and_mul_kernel` (model-only) | **S** |
| **swiglu_and_mul** `activation.py:swiglu_and_mul` | `test_activation.py` (swiglu path) | bf16 swiglu | **DONE** `swiglu_and_mul_kernel` (model-only) | **S** |
| **gelu_and_mul / gelu_tanh_and_mul** `activation.py:gelu_and_mul` | `test_activation.py:torch_gelu_ref` | bf16 gelu·mul | **DONE** `gelu_and_mul_kernel`, `gelu_tanh_and_mul_kernel` (model-only) | **S** |
| **silu_and_mul + quant** `activation.py:silu_and_mul_quant`; flydsl `test_silu_and_mul_fq.py` | `flydsl_tests/test_silu_and_mul_fq.py:_swiglu_ref`, `test_activation.py:_ref_group_scales_fp4/fp8` | silu·mul → fp8/fp4 group quant | **DONE** `silu_and_mul_quant_kernel` (model-only) | M |
| **scaled_silu_and_mul / *_bias** `activation.py:scaled_silu_and_mul, *_and_mul_bias` | `test_activation.py` | scaled/biased act·mul | not-yet | **S** |
| **gelu_fast** `activation.py:gelu_fast` | `test_activation.py:torch_gelu_ref` | tanh-approx gelu | **DONE** `gelu_fast_kernel` (model-only) | **S** |

---

## 8. KV-cache

| op / API | torch ref | AMD semantics | models | status | difficulty |
|---|---|---|---|---|---|
| **reshape_and_cache (+flash)** `cache.py:reshape_and_cache(_flash)` | `test_kvcache.py` | write K/V into paged cache, bf16 | all serving | not-yet | M (layout) |
| **reshape_and_cache + per-token/block quant** `cache.py:reshape_and_cache_with_{pertoken,block}_quant` | `test_kvcache.py`, `test_kvcache_blockscale.py` | fp8 KV cache quant on write | fp8 serving | not-yet | M |
| **concat_and_cache_mla** `cache.py:concat_and_cache_mla` | `test_concat_cache_mla.py` | MLA compressed-KV cache write | **DeepSeek** MLA | not-yet | M |
| **indexer k/qk quant + cache** `cache.py:indexer_k_quant_and_cache`, `indexer_qk_rope_quant_and_cache` | `test_indexer_k_quant_and_cache.py` | DS indexer fp8 cache | DeepSeek indexer | not-yet | L |
| **fused qk_rope concat+cache (mla)** `cache.py:fused_qk_rope_concat_and_cache_mla`; triton `fused_kv_cache.py` | (triton fused) | rope + concat + cache fused | DS MLA | **skip** (overlaps MLA/qk-rope; build later) | L |
| **swap/copy blocks** `cache.py:{swap_blocks, copy_blocks}` | `test_kvcache.py` | block movement (host-orchestrated) | serving | **skip** (memory mgmt, not compute) | XL |
| **cp_gather_indexer_k_quant_cache** `cache.py` | `test_indexer_k_quant_and_cache.py` | gather + quant cache | DS indexer | **skip** (niche, overlaps indexer) | L |

---

## 9. Sampling / Top-k

| op / API | torch ref | AMD semantics | status | difficulty |
|---|---|---|---|---|
| **top_k_top_p sampling from probs** `sampling.py:top_k_top_p_sampling_from_probs` (+`top_p_*`, `top_k_renorm_probs`) | `test_sampling.py` | top-k/top-p renorm + categorical sample | not-yet | M (randomness → seed-pin) |
| **greedy / random / mixed sample** `sample.py:{greedy_sample, random_sample, mixed_sample}` | `test_sample.py` | argmax / exponential-trick sampling | not-yet | M |
| **top_k_per_row (prefill/decode)** `topk.py:top_k_per_row_{prefill,decode}(_fast)` | `test_topk_per_row.py`, `test_topk_row_prefill.py` | per-row top-k selection | not-yet | S/M |
| **topk_plain** `topk_plain.py` | `test_topk_plain.py` | plain top-k | not-yet | **S** |
| **moe_fused_gate** `topk.py:moe_fused_gate` | `test_moe_topk_gating.py` | fused gate+topk (see MoE §2) | not-yet (dup of MoE topk) | M |
| **exponential / outer_exponential** `sample.py:exponential` | `test_sample.py` | RNG primitive | **skip** (stochastic primitive) | XL |

---

## 10. Other

| op / API | torch ref | AMD semantics | models | status | difficulty |
|---|---|---|---|---|---|
| **jagged_dense_bmm + broadcast add** flydsl / triton | (task model.py) | jagged-batched bmm + bias broadcast | rec models | **DONE** `jagged_dense_bmm_kernel` (torch2flydsl, inlined) + `triton2flydsl/generative_recommenders/jagged_dense_bmm_broadcast_add` (triton source) | M |
| **gated delta rule (linear attn)** `chunk_gated_delta_rule_fwd_h.py`; triton `gated_delta_net/gated_delta_rule.py`; flydsl `linear_attention_kernels.py` | `test_gated_delta_rule.py:{recurrent,chunk}_gated_delta_rule_ref`; `flydsl_tests/...:ref_chunk_gated_delta_rule_fwd_h` | chunked gated delta-rule recurrence | **Qwen3-Next**, MiniMax (linear attn) | not-yet (flydsl backend exists → good fit) | L (stateful recurrence) |
| **causal conv1d (+update)** `causal_conv1d.py`; triton `conv/causal_conv1d*` | `triton_tests/conv/test_causal_conv1d.py:causal_conv1d_ref(_update)` | depthwise causal conv (SSM/Mamba) | Mamba/hybrid | not-yet | M (stateful update variant) |
| **MHC (multi-head compress / sinkhorn)** `mhc.py`; triton utils `mhc_ref.py` | `test_mhc.py:{mhc_pre_ref, mhc_post_ref}`, `utils/mhc_ref.py:mhc_torch` | sinkhorn-knopp routing/compress | niche | not-yet (complex) | L |
| **grouped matmul (gmm/tgmm) / moe fused gemm** triton `gmm.py:{gmm, ptgmm, nptgmm}`, `moe_op_gemm` | `triton_tests/test_gmm.py:{torch_gmm, torch_tgmm}` | grouped GEMM (ragged) | MoE/serving | **DONE (triton2flydsl)** `aiter/moe_fused_gemm`; plain gmm/tgmm not-yet | M |
| **gather_kv_b_proj** triton `gather_kv_b_proj.py` | `triton_tests/test_gather_kv_b_proj.py:ref_gather_kv_b_proj` | gather + KV b-proj (fp4) | DS MLA | **skip** (niche, fp4 layout-heavy) | L |
| **split_gdr_update / fused rearrange sigmoid gdr** `fused_split_gdr_update.py` | `test_split_gdr_update.py`, `triton_tests/test_fused_rearrange_sigmoid_gdr.py:ref_fused_rearrange_sigmoid_gdr` | gated delta-rule update helpers | linear-attn models | **skip** (overlaps gated-delta-rule) | L |
| **communication / all-reduce** `communication.py`, `custom_all_reduce.py`, `quick_all_reduce.py` | `multigpu_tests/*` | collective comms | distributed | **skip** (multi-GPU) | XL |

---

## Recommended build order / batching

> **Status:** Batches A–D are **built** (activations, norms, rope, standalone
> quantizers, the quantized GEMMs, the MoE routing/gating + dtype variants).
> Batch E attention is built in the **`triton2flydsl`** suite. The remaining
> tail is **Batch F** (MLA/paged/linear-attn/conv as PyTorch-`Model` refs),
> **KV-cache**, **sampling/top-k**, and torch2flydsl attention references.
> Kept below as the original prioritization rationale.

Prioritized so each batch (a) reuses emulation already proven in existing tasks,
(b) front-loads the high-frequency ops that every target model runs, and (c)
defers stateful / metadata-heavy / multi-GPU ops.

### Batch A — elementwise & norm warmups (all `S`, fp/simple refs)
Fast to author, exercise the FlyDSL elementwise/reduction path, broad model coverage.
- `silu_and_mul`, `swiglu_and_mul`, `gelu_and_mul` — ref `op_tests/test_activation.py:torch_silu_and_mul / torch_gelu_ref`.
- `rmsnorm2d` + `fused_add_rmsnorm` — ref `op_tests/test_rmsnorm2d.py` / `triton_tests/normalization/test_fused_rmsnorm_add.py:run_torch`.
- `layernorm2d` — ref `op_tests/test_layernorm2d.py`.
- `rope_fwd` (neox + cached) — ref `op_tests/test_rope.py` / `triton_tests/torch_compile/test_compile_rope.py`.

### Batch B — standalone quantizers (reuse `moe_kernel` mxfp4/e8m0 emulation)
- `quant_mxfp4` (per-1x32 + e8m0) — ref `op_tests/test_quant_mxfp4.py:ref_quant_mxfp4`.
- `dynamic_mxfp8_quant` — ref `op_tests/triton_tests/quant/test_quant_mxfp8.py:torch_mxfp8_quant_from_fp32`.
- `dynamic_per_token_scaled_quant` (fp8) — ref `op_tests/test_quant.py` / `triton_tests/torch_compile/test_compile_quant_per_token.py`.
- `smoothquant_fwd` — ref `op_tests/test_smoothquant.py`.

### Batch C — quantized GEMMs (highest model value; reuse Batch B quant emul + `hgemm`/`moe` GEMM patterns)
- `gemm_a8w8_blockscale` (fp8 128×128) — ref `op_tests/test_gemm_a8w8_blockscale.py:run_torch`; shapes `model_configs/a8w8_blockscale_*tuned_gemm_ds_v3.csv` (**DeepSeek-V3**).
- `gemm_a8w8` (fp8 per-token) — ref `op_tests/test_gemm_a8w8.py:run_torch`.
- `gemm_a4w4` (mxfp4) — ref `op_tests/test_gemm_a4w4.py:run_torch`; shapes `dsv3_a4w4_blockscale_*`.
- `gemm_a8w8_bpreshuffle` (fp8 + preshuffle) — ref `op_tests/test_gemm_a8w8_bpreshuffle_pad_k.py`; shapes `dsv3_a8w8_bpreshuffle_*`, `glm47`.
- `batched_gemm_bf16` / `batched_gemm_a8w8` — refs `op_tests/test_batched_gemm_*.py:run_torch`.

### Batch D — MoE routing/gating + remaining MoE GEMM dtype
- `topk_gating` (softmax top-k renorm) — ref `op_tests/test_moe_topk_gating.py:run_torch_softmax`.
- `biased_grouped_topk` (DeepSeek routing) — ref ships in `topk.py:biased_grouped_topk_torch`.
- `fmoe_fp8_blockscale_g1u1` (MoE a8w8 blockscale) — ref `op_tests/test_moe_blockscale.py`; shapes `a8w8_blockscale_*fmoe_ds_v3.csv`.

### Batch E — attention (medium): the dense + sink path
- `flash_attn_func` (bf16 MHA) — ref `op_tests/test_mha.py:run_torch`.
- `flash_attn_varlen_func` — ref `op_tests/test_mha_flydsl_varlen.py:_ref_mha_varlen` (FlyDSL varlen ref already exists → good fit).
- `fmha_fwd_with_sink_asm` (**GPT-OSS** attention sink) — ref `op_tests/test_fmha_fwd_with_sink_asm.py:_ref_attn`.

### Batch F — harder / stateful (do last, one at a time)
- MLA decode/prefill (`test_mla.py:torch_mla_extend`) — **DeepSeek**, but metadata + split-kv reduce = `L`.
- paged attention (`test_pa_ragged.py:run_torch`) — block tables = `L`.
- gated delta rule / linear attention (`test_gated_delta_rule.py`) — Qwen3-Next; FlyDSL `linear_attention_kernels.py` exists, so the backend is reachable, but recurrence is `L`.
- causal_conv1d (`test_causal_conv1d.py:causal_conv1d_ref`) — Mamba/hybrid.

### Poor fits — do NOT build (flag)
- **Backward / training** ops: `mha_bwd`, `fmha_v3_bwd`, `rope_bwd` (inference kit).
- **Gluon-language** kernels: `pa_decode_gluon`, gluon `mla_decode`, gluon `pa_mqa_logits`, sage attention (`fav3_sage*`) — Triton-Gluon dependency, no clean FlyDSL target.
- **gfx1250-only**: `flydsl/bpreshuffle_gemm_gfx1250.py`, `flydsl/grouped_moe_gfx1250.py`, `test_flydsl_grouped_gemm_gfx1250.py` — wrong arch for the gfx942/gfx950 target.
- **Multi-GPU / distributed**: `communication.py`, `custom_all_reduce`, `quick_all_reduce`, `test_moe_ep/dp_share/local_expert`, `multigpu_tests/*`.
- **Infra / tuning, not ops**: `gemm_*_tune`, `test_gemm_codegen.py`, `test_pretune.py`, `gradlib`.
- **Memory-management, not compute**: `cache.py:{swap_blocks, copy_blocks}`.
- **Stochastic primitives** (hard to gate deterministically): `sample.py:exponential`, raw `random_sample` (samplers usable only with pinned seeds).

---

## Notes & caveats

- **(b)-faithfulness reuse**: the mxfp4/e8m0 numerics emulation in
  `tasks/torch2flydsl/moe_kernel/model.py` (`_f32_to_e2m1_codes`, `_f32_to_e8m0`,
  `_mxfp4_dequant`) is the canonical, bit-exact AMD quantizer — copy it into the
  fp4 GEMM/quant tasks rather than re-deriving.
- Many ops have **CK / ASM / Triton / FlyDSL** backends simultaneously. The
  `model.py` only needs the PyTorch (b)-faithful reference; the FlyDSL target is
  the porting work. Ops where a **FlyDSL backend already exists**
  (`ops/flydsl/`: gemm, fmha, moe, linear attention, moe_sorting) are lower-risk
  because the device-kernel target is known to compile.
- Some "ops" are actually **fused pipelines** (e.g. `fused_qk_norm_rope_cache_quant`
  combines rmsnorm+rope+quant+cache). Cataloged as one op; their building blocks
  (rope, rmsnorm, quant) are also listed separately and are better first tasks.
