# Plan / record: the `torch2flydsl` task type in AgentKernelArena

**Status: DELIVERED.** The `torch2flydsl` task type exists, its contract is
settled in [`TORCH2FLYDSL_CONVENTIONS.md`](TORCH2FLYDSL_CONVENTIONS.md), the op
surface + per-task mapping is in [`AITER_OPS_CATALOG.md`](AITER_OPS_CATALOG.md),
and 48 `torch2flydsl` + 10 `triton2flydsl` task dirs are built on branch
`torch2flydsl`. For continuing the work on another node read
[`HANDOFF_GFX942.md`](HANDOFF_GFX942.md).

This file is the condensed design record: what was decided, why, and the
per-phase changelog. The verbose phase-by-phase prose was removed once the
CONVENTIONS contract and the CATALOG built-index made it redundant.

---

## 1. What it is

A task type whose **source is PyTorch in KernelBench format**
(`class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`) and whose
**target is a FlyDSL kernel**. It fuses the `torch2hip` input layer (a `Model`
PyTorch file) with the `flydsl2flydsl` target/harness layer (`kernel.py` +
`test_kernel_harness.py`). Task discovery is automatic: `src/tasks.py` globs
`tasks/**/config.yaml`, so dropping a `tasks/torch2flydsl/<op>_kernel/config.yaml`
registers the task — no engine edit. The cheatsheet resolver derives the target
language from the name (`"torch2flydsl".split('2')[-1] == "flydsl"`).

Sibling suite **`triton2flydsl`**: standalone Triton kernel (aiter/Meta) as the
SOURCE/reference, no torch reference, FlyDSL is GEAK's target. Same harness/config
machinery.

## 2. Resolved design decisions (these were the original open questions)

1. **Task-type name** = `torch2flydsl` (matches the `torch*` source spelling used
   by `torch2hip`; the git *branch* is `torch2flydsl`). The earlier branch name
   `pytorch2flydsl` was abandoned.
2. **Engine registration** is minimal: add `torch2flydsl_task_type()` to
   `src/prompts/task_type.py` and one `elif` in `src/prompt_builder.py`. No
   cheatsheet/Makefile/discovery edits (FlyDSL plumbing already exists).
3. **Correctness fidelity = option (b), AMD-runtime-faithful.** `model.py`
   reproduces the real quant/dtype/layout semantics the op runs with on AMD via
   aiter (validated against the *real aiter op* as ground truth), NOT an
   idealized fp32 op. The harness gates at a **tight** bound (≈1 output-dtype ULP;
   `max|ref-out|/max|ref| ≤ 1e-2`), never a loosened cosine/pass-% band. The
   early idea of comparing an mxfp4 kernel to an *unquantized* bf16 reference was
   rejected (only ~20% close — inherent 4-bit error); the reference is quantized
   to match the kernel apple-to-apple. One documented exception:
   `fmoe_g1u1_tkw1_kernel` `TOL=3.5e-2` (fp8-MFMA floor — see CONVENTIONS §4).
4. **`kernel.py` shipped IFF a clean standalone FlyDSL device kernel exists**
   (§6 rule). 8 tasks inline a real kernel; the other 40 are model-only (FlyDSL
   is GEAK's target). Do not fork an unrelated kernel just to have a `kernel.py`.
5. **`get_init_inputs()` is a FLAT positional list** (`Model(*get_init_inputs())`),
   NOT the `torch2hip` `[args_list, kwargs_dict]` form.
6. **No `GEAK_*` markers** in harnesses — AKA's evaluator runs the commands with
   `cwd=workspace` and parses `build/performance_report.json`; the kernel dir
   resolves from the file/cwd.
7. **One-dir-per-op** (harness enumerates real shapes from `configs/*.csv`
   internally), not one-dir-per-shape.

## 3. AKA branch-state FLAG (still open)

`git reflog` showed a `pytorch2flydsl → torch2flydsl` checkout that **no agent
issued** (no branch/checkout command was run by any worker). Both branches
pointed at the same commit so no work was lost, but the unexplained switch is
flagged for the maintainer. All work since has been on `torch2flydsl`, and the
bulk of it is still **uncommitted** (see HANDOFF §C).

## 4. Per-phase changelog (what was built, in order)

All validated on gfx950 (flydsl 0.1.9). Each task: tight gate ALL PASS,
`--full-benchmark` exit 0; all files byte-compile; `grep -i geak|judge` = 0; no
commits, no branch switches.

| Phase | Tasks added / reworked | Notes |
|---|---|---|
| 1–3 | `hgemm_kernel`, `moe_kernel` | hgemm + MoE a4w4 reworked to **self-contained inline** FlyDSL `kernel.py` (no aiter device-kernel import; only host-side quant/shuffle/sorting prep allowed). MoE inlines the full a4w4 two-stage device kernel (~6.6k lines, 5 merged modules; only 2 name collisions resolved: `crd2idx`→`_pre_crd2idx`, `_if_then`→`_epi_if_then`). |
| 4 | `qk_norm_rope_quant_kernel`, `moe_a8w4_kernel`, `moe_swiglu_kernel` | qk-norm+GPT-J-rope(+optional fp8 quant) inlined; a8w4 (fp8 act/fp4 weight) and swiglu (fp4) reuse the same inline MoE kernel via `a_dtype`/`act` args. |
| 5 | `jagged_dense_bmm_kernel` | jdbba clean layout-API prototype inlined (source read without checkout from aiter `origin/anguyenh/flydsl-jdbba`). |
| 6 | `moe_sorting_kernel` | integer counting-sort, **EXACT** bit-for-bit gate; oneshot path inlined (2k/4k HBM multiphase deferred); cross-checked vs `aiter.fused_moe.moe_sorting` 6/6. |
| 7+ | the remaining 40 model-only torch2flydsl tasks + the 10 triton2flydsl tasks | activations, norms, rope, standalone quantizers, the quantized GEMMs, MoE routing/gating + dtype variants (model-only); attention/MLA/sage/logits + grouped-gemm + jdbba-broadcast as triton2flydsl. See the CATALOG built-index for the full dir→op map. |

## 5. Key precedents / files relied on (exact paths)

- Input format: `tasks/torch2hip/gpumode/3267_SimpleMatmulModule/*`,
  `tasks/torch2hip/gpumode/1178_MLP_model/*`.
- FlyDSL target/harness: `tasks/flydsl2flydsl/hgemm_splitk_kernel/*`.
- Engine: `src/tasks.py`, `src/prompt_builder.py`, `src/prompts/task_type.py`,
  `src/prompts/cheatsheet/default_cheatsheet.yaml`, `src/evaluator.py`,
  `src/performance.py`, `main.py`, `Makefile`, root `config.yaml`.
- Registration precedent: PR #39 (`flydsl2flydsl`), merge `98b17d6`.
- Op sources + real shapes: `aiter/aiter/ops/{flydsl,triton}/...`,
  `aiter/op_tests/...`, `aiter/configs/` + `aiter/configs/model_configs/*.csv`.
- Canonical AMD-faithful mxfp4/e8m0 emulation:
  `tasks/torch2flydsl/moe_kernel/model.py` (`_f32_to_e2m1_codes`, `_f32_to_e8m0`,
  `_e8m0_to_f32`, `_mxfp4_dequant`) — copy, don't re-derive.
