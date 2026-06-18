# HANDOFF — continue the `torch2flydsl` corpus on a gfx942 node

Self-contained continuation brief for a fresh agent (an "exact copy of me")
resuming this exact work on a **gfx942** (CDNA3, MI300/MI325) node. Read this +
[`TORCH2FLYDSL_CONVENTIONS.md`](TORCH2FLYDSL_CONVENTIONS.md) (the contract) +
[`AITER_OPS_CATALOG.md`](AITER_OPS_CATALOG.md) (op surface + built-task index)
and you can resume without losing context. This doc cross-references those rather
than duplicating them.

> **The single most important thing first:** all this work is **UNCOMMITTED** on
> branch `torch2flydsl` and `git push` to the canonical remote is **blocked**. If
> the gfx942 node is a different machine, you must commit + push/transfer first or
> none of it exists there. See **§C**.

---

## A. Project goal & framing

We are building a **corpus of GPU-kernel translation tasks** in AgentKernelArena
(AKA) for **GEAK** (the kernel-authoring agent, used by "Hyperloom") to author
**FlyDSL** kernels for the PyTorch/Triton op APIs that real models actually run on
AMD via **aiter**. Target models: DeepSeek-V3/V4, GPT-OSS, Llama, Qwen3, Kimi,
GLM, MiniMax. Each task gives GEAK a faithful reference + a correctness/perf
harness; GEAK produces (or optimizes) a FlyDSL `kernel.py`.

Two suites (both auto-discovered by `src/tasks.py` globbing `tasks/**/config.yaml`;
both follow the conventions doc; tasks live under `tasks/<type>/...` with a
`config.yaml` declaring the `--compile` / `--correctness` / `--full-benchmark`
commands):

- **`torch2flydsl`** — a PyTorch `class Model` in **KernelBench format**
  (`Model` / `forward` / `get_inputs` / `get_init_inputs`) is the **SOURCE**.
  Fidelity is **option (b), AMD-runtime-faithful**: the `model.py` reference
  reproduces aiter's *real* quant/dtype/layout semantics (validated against the
  real aiter op as ground truth), not an idealized fp32 op. A FlyDSL `kernel.py`
  is shipped **only when a clean standalone FlyDSL device kernel exists** (§6 of
  CONVENTIONS); otherwise FlyDSL is GEAK's target and the task is model-only.
- **`triton2flydsl`** — a **standalone Triton kernel** (from aiter/Meta) is the
  SOURCE/reference (triton + torch only, faithful to upstream, **no torch
  reference**); FlyDSL is GEAK's target, added later. The harness runs the
  **Triton** kernel.

## B. Current state (verified)

Verified on the current node (gfx950) on `2026-06-19`:

| Item | Value |
|---|---|
| AKA repo | `/workspaces/WS/AgentKernelArena` |
| Branch | `torch2flydsl` |
| aiter repo | `/workspaces/WS/aiter`, HEAD `e773193041689602e5431e110ef39db0390e7015` |
| flydsl version | `0.1.9` (`python3 -c "import flydsl; print(flydsl.__version__)"`) |
| GPU arch (this node) | **gfx950** (CDNA4); target node is **gfx942** (CDNA3) |
| Task dirs | **48 `torch2flydsl` + 10 `triton2flydsl` = 58** |

### Task inventory by suite/category

**`torch2flydsl` — 48 dirs.** 8 ship an inlined FlyDSL `kernel.py` (apple-to-apple
validated); 40 are model-only (FlyDSL = GEAK's target):

- **Inlined `kernel.py` (8):** `hgemm_kernel`, `gemm_a8w8_bpreshuffle_kernel`,
  `moe_kernel` (a4w4), `moe_a8w4_kernel`, `moe_swiglu_kernel`,
  `moe_sorting_kernel`, `qk_norm_rope_quant_kernel`, `jagged_dense_bmm_kernel`.
- **GEMM model-only (11):** `gemm_a8w8`, `gemm_a8w8_blockscale`, `gemm_a4w4`,
  `gemm_a16w8_blockscale`, `gemm_a8w8_per_token_scale`, `gemm_a16wfp4`,
  `gemm_a8wfp4`, `gemm_afp4wfp4`, `gemm_afp8wfp8`, `batched_gemm_bf16`,
  `batched_gemm_a8w8`.
- **MoE model-only (7):** `fmoe_fp8_blockscale_g1u1`, `fmoe_g1u1_tkw1`,
  `moe_2stage_generic`, `moe_topk_softmax`, `moe_topk_sigmoid`,
  `moe_topk_softplus`, `moe_biased_grouped_topk`.
- **Norm (6):** `rmsnorm2d`, `fused_add_rmsnorm`, `rmsnorm2d_dynamicquant`,
  `rmsnorm2d_smoothquant`, `layernorm2d`, `layernorm2d_with_add`.
- **RoPE (3):** `rope_fwd`, `rope_2d_fwd`, `rope_thd_fwd`.
- **Quant (7):** `quant_mxfp4`, `dynamic_mxfp8_quant`, `per_token_fp8_quant`,
  `per_tensor_fp8_quant`, `per_1x128_fp8_quant`, `per_token_i8_quant`,
  `smoothquant`.
- **Activation (6):** `silu_and_mul`, `swiglu_and_mul`, `gelu_and_mul`,
  `gelu_tanh_and_mul`, `gelu_fast`, `silu_and_mul_quant`.

(All `*_kernel` suffixed on disk, e.g. `tasks/torch2flydsl/gemm_a8w8_kernel/`.)

**`triton2flydsl` — 10 dirs** (each: `<name>.py` Triton/torch source +
`test_kernel_harness.py` + `config.yaml`; no torch ref, no `kernel.py`):
`aiter/mha`, `aiter/mla`, `aiter/fav3_sage`, `aiter/fav3_sage_mxfp4`,
`aiter/fp8_mqa_logits`, `aiter/unified_attention`,
`aiter/unified_attention_sparse_mla`, `aiter/moe_fused_gemm`,
`aiter/moe_routing_sigmoid_top1`,
`generative_recommenders/jagged_dense_bmm_broadcast_add`.

### Uncommitted status (`git status --short`)

- **Modified (tracked):** `hgemm_kernel/test_kernel_harness.py`,
  `jagged_dense_bmm_kernel/{model.py,test_kernel_harness.py}`,
  `qk_norm_rope_quant_kernel/{model.py,test_kernel_harness.py}`.
- **Untracked:** the 3 root docs (`AITER_OPS_CATALOG.md`,
  `TORCH2FLYDSL_CONVENTIONS.md`, `TORCH2FLYDSL_PLAN.md` + this `HANDOFF_GFX942.md`)
  **and 41 of the 48 `torch2flydsl` task dirs** (all the model-only batch +
  `gemm_a8w8_bpreshuffle_kernel`). The 7 phase-1–6 inlined tasks and the 10
  `triton2flydsl` dirs are committed (`git log`: `14dda87 jagged_bmm_broadcast`,
  `e9f766b triton kernels`, `9540c81 other ops`, …).

**⇒ The majority of the corpus is untracked. Do not assume it exists on another
checkout. See §C.**

## C. CRITICAL move prerequisite — commit & push BEFORE switching nodes

`origin` is `git@github.com:AMD-AGI/AgentKernelArena.git` — **we have no write
access** (push blocked; needs a fork or collaborator rights). A `fork` remote is
configured (`git@github.com:amd-yashagar/AgentKernelArena.git`), **but its push
has NOT been verified**: an earlier `git push -u fork torch2flydsl` failed with
`ERROR: Repository not found` because the fork repo did not exist on GitHub yet.
Before relying on it, confirm the fork repo actually exists (create it via the
GitHub UI if not).

If the gfx942 node is a **different machine**, the 58 task dirs + these docs
(most of which are uncommitted/untracked) will **not exist there** unless you
commit and push/transfer first. Exact steps:

```bash
cd /workspaces/WS/AgentKernelArena
git status --short                 # confirm what is uncommitted
# build/ and __pycache__ are gitignored, so -A will NOT stage perf-report/cache junk:
git add -A                         # stages the 41 untracked task dirs + docs + the 5 modified files
git commit -m "torch2flydsl: full op corpus (48) + triton2flydsl (10) + handoff docs"

# Push: `fork` is configured to amd-yashagar/AgentKernelArena, but VERIFY it is
# pushable first — an earlier `git push -u fork torch2flydsl` failed with
# "Repository not found" because the fork repo did not exist on GitHub. Create the
# fork on GitHub (or confirm it exists), then:
git push -u fork torch2flydsl
#   If the push fails (fork missing / no access), DO NOT assume it succeeded —
#   transfer the working tree directly instead (rsync/scp, see below).

# On the gfx942 node (only after a confirmed successful push):
git clone -b torch2flydsl git@github.com:amd-yashagar/AgentKernelArena.git
```

If you cannot push at all, **transfer the working tree directly** (rsync/scp the
whole `/workspaces/WS/AgentKernelArena` including untracked dirs). Verify on the
target with the §B counts (48 + 10) before continuing. Do **not** force-push or
rewrite history; do not commit `build/` artifacts or `__pycache__`.

## D. gfx942 portability (the key reason for the move)

From the dtype analysis of the ~58 tasks:

- **~33 run as-is** on gfx942 (bf16 + int8 ops): all norms, rope, activations,
  bf16 GEMMs (`hgemm`, `batched_gemm_bf16`), int8/smoothquant quantizers,
  MoE routing/topk/sorting, `jagged_dense_bmm`, and the bf16 attention
  (`triton2flydsl/aiter/{mha,mla,unified_attention,...}`).
- **~16 run after a small fp8 change.** gfx942 fp8 is **`e4m3fnuz`** (max ≈ 240);
  gfx950 fp8 is **`e4m3fn`** (max = 448). The fp8 references currently **hardcode
  the gfx950 dtype** (`torch.float8_e4m3fn`) and so must become **arch-selected**
  to match the gfx942 aiter op. Affected: `gemm_a8w8`, `gemm_a8w8_blockscale`,
  `gemm_a8w8_bpreshuffle`, `gemm_afp8wfp8`, `gemm_a8w8_per_token_scale`,
  `batched_gemm_a8w8`, `fmoe_fp8_blockscale_g1u1`, `fmoe_g1u1_tkw1`,
  `moe_2stage_generic`, `per_token_fp8_quant`, `per_tensor_fp8_quant`,
  `per_1x128_fp8_quant`, `dynamic_mxfp8_quant`, `rmsnorm2d_dynamicquant`,
  `silu_and_mul_quant`, the fp8 attention paths.
- **9 fp4/mxfp4 tasks CANNOT run on gfx942** (CDNA3 has **no native FP4 MFMA**) —
  **gfx950-only**: `gemm_a4w4`, `moe_kernel` (a4w4), `moe_swiglu_kernel` (a4w4),
  `moe_a8w4_kernel` (fp4 weight), `gemm_a16wfp4`, `gemm_a8wfp4`, `gemm_afp4wfp4`,
  `quant_mxfp4`, and triton `fav3_sage_mxfp4`.

(33 + 16 + 9 = 58.)

**Action items on the gfx942 box, in order:**

1. **Add an arch-selected fp8 dtype helper** to the fp8 references (a single
   helper that returns `torch.float8_e4m3fnuz` on gfx942 / `torch.float8_e4m3fn`
   on gfx950, keyed off the device arch), and route every hardcoded
   `float8_e4m3fn` through it so the reference matches the gfx942 aiter op's fp8
   numerics. Keep the tight gate.
2. **Tag the 9 fp4 tasks `gfx950-only`** in their `config.yaml` (an explicit arch
   marker / skip note) so a gfx942 run does not treat them as failures.
3. **RE-VALIDATE the bf16/int8/fp8 set** (the ~49 non-fp4 tasks) on the gfx942
   box via `python3 test_kernel_harness.py --correctness` at the existing tight
   gate. fp8 tasks must pass against the `e4m3fnuz` reference after step 1.
4. **Confirm flydsl availability/version on gfx942.** For the 8 inlined-kernel
   tasks: `hgemm` has a gfx942 path; the fp4 FlyDSL kernels (`moe_kernel`,
   `moe_swiglu_kernel`, `moe_a8w4_kernel`) are **gfx950-only**.
   `gemm_a8w8_bpreshuffle`, `moe_sorting`, `qk_norm_rope_quant`,
   `jagged_dense_bmm` are fp8/bf16/int and should build on gfx942 (re-validate).

## E. Open flags / decisions

1. **Tolerance exception (needs sign-off):** `fmoe_g1u1_tkw1_kernel` uses
   `TOL=3.5e-2` — the fp8-MFMA accumulation floor (measured cosine ≈ 2.7e-4). It
   is the only task above the 1e-2 bound and is a real hardware-precision floor,
   not a loosened gate. Confirm acceptance.
2. **Upstream aiter bug (report upstream):** `fav3_sage`'s `_sage_fwd_mask`
   omits `USE_BIAS` (the bias path is not wired into the masked variant).
3. **Repo hook returns invalid JSON:** `.cursor/hooks/destructive-block.sh`
   emits invalid JSON, which blocks `rm -rf` cleanups. Harmless leftover
   `__pycache__/` and `build/` dirs accumulate in task folders; they are
   gitignore-able and should not be committed.
4. **Unexplained branch switch:** a checkout `pytorch2flydsl → torch2flydsl`
   happened earlier that **no agent issued** (no branch/checkout command was
   run). Both branches pointed at the same commit so nothing was lost; flagged
   for the maintainer.

## F. Per-op fidelity gotchas to PRESERVE

These are exact AMD-runtime semantics; the references match aiter's real
numerics. **Never loosen a gate to work around them.**

- **mxfp4 e8m0 scale differs by op:** MoE **a4w4** uses **RoundUp / ÷4**; GEMM
  **a4w4** uses **EVEN-mode** `floor(log2) − 2`. Use the right one per op.
- **smoothquant truncates toward zero** (not round-to-nearest-even).
- **Fused norm→quant keeps fp32 internally** (reduce + scale in fp32, quantize at
  the end).
- **`gemm_a16wfp4` prequants its activation to mxfp4 inside the kernel** (the
  reference must mirror that internal prequant).
- Canonical mxfp4/e8m0 emulation lives in `moe_kernel/model.py`
  (`_f32_to_e2m1_codes`, `_f32_to_e8m0`, `_e8m0_to_f32`, `_mxfp4_dequant`) — copy
  it, don't re-derive.

## G. How to continue (the remaining tail)

Read CONVENTIONS (contract) + CATALOG (op surface + built-index). Fan out the
**not-yet** ops in **bounded per-category batches** (cap ~4–5 ops/worker),
authoring each `model.py` to be option-(b) faithful and validating it against the
**real aiter op** at the tight gate. Remaining work:

- **KV-cache:** `reshape_and_cache(_flash)`, `+pertoken/block quant`,
  `concat_and_cache_mla`, indexer k/qk quant+cache.
- **Sampling / top-k:** `top_k_top_p_sampling_from_probs`, greedy/random/mixed
  sample, `top_k_per_row_{prefill,decode}`, `topk_plain` (pin seeds for the
  stochastic ones).
- **torch2flydsl attention references** (PyTorch `Model` as source): SDPA / MHA /
  MLA — the `triton2flydsl` attention suite already exists; these are the
  Model-as-source counterparts.
- **GEMM tail:** preshuffle / deepgemm / asm variants
  (`gemm_a8w8_blockscale_bpreshuffle`, `deepgemm`, a8w8 asm/flatmm).
- **Fused norm/act/rope + smoothquant/fp4 long-tail:** mRoPE cache-quant, qk
  rmsnorm group-quant, gated-rmsnorm fp8 group-quant, fused RMS→mxfp4/mxfp8,
  groupnorm, gated-delta-rule (linear attn), causal_conv1d.

**Excluded (poor fit) — do NOT build:** backward/training (`*_bwd`),
gfx1250-only (nvfp4, gluon, grouped_moe_gfx1250), multi-GPU/distributed
(communication, all-reduce, moe EP/DP-share), infra/tuning (`*_tune`,
`test_gemm_codegen`, gradlib), memory-management (`swap_blocks`/`copy_blocks`),
raw stochastic primitives (`exponential`).

## H. Env / setup

- **aiter** at `/workspaces/WS/aiter`, HEAD `e773193041…` — pin/note the commit;
  `config.yaml` `source_origin.commit` records what each kernel was inlined from.
- **flydsl** version `0.1.9` (install/verify with `make setup-flydsl` /
  `make verify-flydsl` from the AKA repo root; confirm the gfx942 box has a
  matching flydsl with a gfx942 codepath).
- **AKA** at `/workspaces/WS/AgentKernelArena`.
- **Running a task** (matches what AKA's evaluator runs, `cwd` = the task dir):
  ```bash
  cd tasks/torch2flydsl/<op>_kernel
  python3 -c "import model; print('compile ok')"     # or the config's build_* compile_command
  python3 test_kernel_harness.py --correctness       # exit 0 = ALL PASS, exit 1 = any fail
  python3 test_kernel_harness.py --full-benchmark     # writes build/performance_report.json
  ```
- **aiter JIT-compiles CK on first call** — the first `--correctness` for a CK/
  quant op is slow (tens of seconds), then cached. Budget for it; it is within
  AKA's 3600s command timeout.
- Harnesses for heavy/shared-GPU tasks include a **transient-OOM retry/backoff**
  wrapper (`_retry(...)` on "out of memory"/"hip" errors). Check `gpu-usage`
  before benchmarking on a shared box.

---

*Cross-references: [`TORCH2FLYDSL_CONVENTIONS.md`](TORCH2FLYDSL_CONVENTIONS.md)
(the per-task contract — file layout, model.py/config/harness/gate, §6 rule),
[`AITER_OPS_CATALOG.md`](AITER_OPS_CATALOG.md) (full op surface + dir→op
built-index + not-yet/skip lists), [`TORCH2FLYDSL_PLAN.md`](TORCH2FLYDSL_PLAN.md)
(condensed design record + resolved decisions + per-phase changelog).*
