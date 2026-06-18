# torch2flydsl — canonical task convention

**Status: this is the authoritative spec that ALL `torch2flydsl` tasks must
follow** (the 48 built tasks under `tasks/torch2flydsl/` + the remaining not-yet
ops in [`AITER_OPS_CATALOG.md`](AITER_OPS_CATALOG.md)). It was derived by
auditing the existing tasks and reconciling their drift. The plan and per-task
implementation notes live in
[`TORCH2FLYDSL_PLAN.md`](TORCH2FLYDSL_PLAN.md); this file is the *contract*.

> **Sibling suite `triton2flydsl`** (10 dirs) uses a *standalone Triton kernel*
> (aiter/Meta) as the SOURCE/reference — triton/torch only, faithful to upstream,
> **no torch reference** — with FlyDSL as GEAK's target (added later). Its tasks
> ship `<name>.py` (the Triton source) + `test_kernel_harness.py` + `config.yaml`
> (`task_type: triton2flydsl`) and the harness runs the **Triton** kernel. The
> §2 PyTorch-`Model` contract below is **torch2flydsl-specific**; everything else
> (config schema, harness CLI/gate/exit contract, comment/SPDX rules) applies to
> both suites.

A `torch2flydsl` task gives a graders a PyTorch reference (`model.py`, KernelBench
format) and asks an agent (GEAK) to author/optimize a FlyDSL kernel that matches
it. Task discovery is automatic: `src/tasks.py` globs `tasks/**/config.yaml`, so
dropping a new `tasks/torch2flydsl/<op>_kernel/config.yaml` registers the task —
no engine edit. The task-type string `torch2flydsl` resolves the FlyDSL
cheatsheet automatically (`target_language = "torch2flydsl".split('2')[-1]`).

---

## 1. File layout / naming (REQUIRED, identical for every task)

A **flat** directory `tasks/torch2flydsl/<op>_kernel/` containing exactly:

```
tasks/torch2flydsl/<op>_kernel/
  model.py                 # PyTorch reference (KernelBench Model) — the SOURCE
  config.yaml              # task descriptor (task_type: torch2flydsl)
  test_kernel_harness.py   # build / correctness / perf harness
  kernel.py                # FlyDSL target — INCLUDE ONLY per §6 rule
  build/                   # created by the harness (performance_report.json); gitignore-able
```

- Directory name: `<op>_kernel` (snake_case op name + `_kernel` suffix), e.g.
  `hgemm_kernel`, `moe_swiglu_kernel`, `rmsnorm2d_kernel`.
- One directory per op (the harness sweeps multiple real shapes internally). Use
  one-dir-per-shape only if per-shape tiling validity forces it.
- Do **not** add `README.md`, `__init__.py`, or extra files per task.

---

## 2. `model.py` — the PyTorch SOURCE (KernelBench contract)

Every `model.py` MUST:

1. **Header (line 1, exact):**
   `# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.`
   followed by a module docstring describing the op math + the AMD-runtime
   semantics it emulates (dtypes, quant, layout).
2. **Be pure-torch.** `import torch` / `torch.nn` / `torch.nn.functional` and the
   stdlib only. **ZERO `aiter` and ZERO `flydsl` imports** (verified by grep over
   all 7). Quant emulation is hand-rolled (see §2.1).
3. Define **`class Model(nn.Module)`** with:
   - `__init__(self, <flat positional hyperparams>)` storing config (and any
     persistent weights as `nn.Parameter` / `nn.Linear`).
   - `forward(self, <runtime tensors>)` computing the op. Math is done in fp32
     accumulation and truncated to the deployed output dtype (usually bf16).
4. **`get_inputs()`** → a flat `list` of the positional tensors passed to
   `forward()` (or a generator yielding such lists for multi-shape). Tensors are
   created on **CPU** (no `device=`/`.cuda()`); the consumer/harness relocates to
   GPU. *(KernelBench convention, matches `tasks/torch2hip/*`.)*
5. **`get_init_inputs()`** → a **FLAT POSITIONAL `list`** such that
   `Model(*get_init_inputs())` constructs the module. Return `[]` when `__init__`
   takes no args. **Do NOT** use the torch2hip `[args_list, kwargs_dict]` form.
6. **No AI/process narration** in comments (no "GEAK", "judge", "let me", "for
   now", "TODO/FIXME", "workaround"); only clean, professional docstrings and
   intent comments. No comment that merely restates the code.

Optional module-level helpers (quantizers, `route_topk`, `_build_cos_sin`,
`_gen_topk`) are allowed and may be imported by the harness; give any
device-taking helper a **`device="cpu"` default** (the harness passes an explicit
device when it needs GPU).

### 2.1 Option-(b) AMD-runtime fidelity (REQUIRED where applicable)

The reference must reproduce the **real quant/dtype/layout semantics the op runs
with on AMD via aiter**, not an idealized fp32 op:

- **MXFP4 (e2m1 + e8m0 per-1×32 block scale):** copy the bit-exact quantizer from
  `tasks/torch2flydsl/moe_kernel/model.py`
  (`_f32_to_e2m1_codes`, `_f32_to_e8m0`, `_e8m0_to_f32`, `_mxfp4_dequant`). Do not
  re-derive it.
- **MXFP8 (e4m3 + e8m0):** see `moe_a8w4_kernel/model.py:_mxfp8_dequant`.
- **bf16 ops** (norm/rope/activation): fp32 reduce/compute, bf16 truncate
  (`qk_norm_rope_quant_kernel`, `hgemm_kernel`).
- **Integer ops** (sorting): reproduce the exact CK/AITER packed layout
  bit-for-bit (`moe_sorting_kernel`).

The reference's numerics define correctness; the harness gate must be tight
against them (§4).

---

## 3. `config.yaml` — task descriptor (REQUIRED schema)

Exact field set and order (all 7 conform):

```yaml
task_type: torch2flydsl
source_file_path:
  - model.py
harness_path: test_kernel_harness.py
compile_command:
  - python3 -c "import torch; from kernel import <builder>; <builder>(<real shape args>); print('compile ok')"
correctness_command:
  - python3 test_kernel_harness.py --correctness
performance_command:
  - python3 test_kernel_harness.py --full-benchmark
target_kernel_functions:
  - <public entrypoint>         # e.g. flydsl_<op>
  - <module builder(s)>         # e.g. build_<op>_module
  - <inner compile/kernel fn>   # optional
source_origin:
  repo: AITER (local)           # or "FlyDSL (local)"
  commit: <40-char sha>
  path:
    - <relative path(s) to the upstream kernel source>
  note: >-
    One paragraph: what model.py references, what kernel.py inlines, what stays
    host-side prep, and the real shape source (configs/*.csv).
prompt:
  source_code: null             # null => prompt_builder lists source_file_path + targets
  instructions: |
    Translate the PyTorch `Model.forward` in model.py into a FlyDSL kernel in
    kernel.py that computes the same result: <one-paragraph op spec>.
    Expose `<entrypoint>(...)` as the entry point. Keep the kernel in FlyDSL —
    do NOT rewrite it in HIP, CUDA, or Triton.
    Correctness and performance are checked with:
      python3 test_kernel_harness.py --correctness
      python3 test_kernel_harness.py --full-benchmark
  task_type: null
  cheatsheet: null
task_result_template: null
```

Rules:
- The CLI verbs are **`--correctness`** and **`--full-benchmark`** (NOT
  `--compile`; FlyDSL JIT-compiles inside the harness). `compile_command` is a
  one-liner that actually **builds the inline kernel** via its module builder and
  prints `compile ok` (so a broken kernel fails the compile phase).
- `source_origin.commit` pins the exact upstream sha the kernel was inlined from.
- `branch:` is added under `source_origin` only when the source is on a non-default
  branch (e.g. `jagged_dense_bmm_kernel`).
- For a **model-only task** (§6, no `kernel.py`): set `compile_command` to a
  `python3 -c "import model; print('compile ok')"` import check, point
  `target_kernel_functions` at the FlyDSL entrypoint name GEAK must produce, and
  word `prompt.instructions` so the agent knows it is authoring `kernel.py` from
  scratch.

---

## 4. `test_kernel_harness.py` — CLI, gate, determinism (REQUIRED shape)

Every harness shares this skeleton (verified across all 7):

- **Header:** `#!/usr/bin/env python3` then the AMD copyright line, then a
  docstring documenting the modes and the gate.
- **Constants:** `KERNEL_FILE = "kernel.py"`, `MODEL_FILE = "model.py"`.
- **Loaders:** `_resolve_kernel_dir()` (prefer the file's dir, fall back to cwd)
  and `_load_module(kernel_dir, filename, alias)` (load by path, honoring the
  workspace dir). Copy verbatim — do not reinvent.
- **`SHAPES`:** a module-level list of dicts, each a named real shape from the
  op's `configs/*.csv` / op_test enum. Include edge cases (unaligned, empty
  groups, M>tile, small/large) where meaningful.
- **`SEED`:** a fixed integer; build inputs with a seeded
  `torch.Generator(device).manual_seed(SEED)` (or `torch.manual_seed(SEED)` +
  `torch.cuda.manual_seed_all(SEED)` for module-weight tasks). Inputs are built on
  **cuda** inside the harness via a `_make_inputs(...)` (the harness owns the GPU
  inputs; `model.get_inputs()` is the CPU KernelBench contract and is only used
  for an optional end-to-end smoke check).
- **Determinism:** any kernel mode with order-dependent fp32 atomics must use a
  **deterministic reduce** path for the correctness comparison (see
  `moe_kernel`: `CORRECTNESS_MODE = "reduce"` vs benchmark `MODE = "atomic"`).
  Never compare against a nondeterministic kernel output.

### Modes

- **`run_correctness(verbose=True)`** — load kernel+model, build the reference via
  the pure-torch `Model` over the harness inputs, run the FlyDSL kernel over the
  SAME inputs/routing, compare with the **tight gate**, print per-shape
  `PASS/FAIL`, a `Status: ALL PASS|FAILED` line, and `correctness: pass|fail`.
  It MUST `assert not failures, ...` on any failure (this is the canonical exit
  contract — §4.1).
- **`run_benchmark(warmup, iters, verbose=True)`** — CUDA-event median timing of
  the kernel and a torch baseline per shape, write
  `build/performance_report.json` as a list of
  `{"test_case_id", "execution_time_ms", "shape", "params", [optional "tflops"/"gbps"]}`,
  and print geomean latency + speedup. `execution_time_ms` is the only field AKA
  parses; the extra metric is informational.

### Tight correctness gate (NEVER loosen)

Pick the gate by op class; all are normalized to the bf16/output precision floor:

| Op class | Gate |
|---|---|
| bf16 GEMM / elementwise / norm / rope | element-wise `isclose(atol=1e-2, rtol=1e-2)` ≥ 99.9% **or** normalized worst-element `max\|ref-out\| / max\|ref\| ≤ 1e-2` |
| quantized (mxfp4 / mxfp8) fused MoE | normalized worst-element `max\|ref-out\| / max\|ref\| ≤ 1e-2` (compare to the **quantized** reference, not an fp32 ideal) |
| integer (sorting) | **EXACT** bit-for-bit equality (no tolerance band) |

The bound is ≈ 1 output-dtype ULP at the output magnitude. **Do not widen
tolerances or use a cosine/pass-% band to force a pass.** If a quantized op can't
hit 1e-2 vs a faithful quantized reference, the reference is wrong, not the gate.

> **One documented exception (needs sign-off):** `fmoe_g1u1_tkw1_kernel` uses
> `TOL=3.5e-2`. This is the fp8-MFMA accumulation floor for that op (measured
> cosine ≈ 2.7e-4); it is a real hardware-precision floor, not a loosened gate.
> It is the only task above the 1e-2 bound and is flagged in `HANDOFF_GFX942.md`
> §E. Do not introduce new exceptions without the same evidence + sign-off.

### 4.1 Canonical `__main__` exit contract (REQUIRED, identical)

```python
if args.correctness:
    try:
        run_correctness()
    except AssertionError as exc:
        print(f"ASSERTION: {exc}")
        sys.exit(1)
    sys.exit(0)
else:
    run_benchmark(warmup=args.warmup, iters=args.iterations)
```

Argparse declares `--correctness`, `--benchmark`, `--full-benchmark`,
`--warmup`, `--iterations`. `--full-benchmark`/`--benchmark` both fall through to
`run_benchmark`. Exit 0 = all-pass, exit 1 = any failure (AKA's evaluator keys on
exit code + absence of "fail").

### 4.2 Optional: transient-OOM retry

For heavy / shared-GPU tasks, a `_retry(fn, tries=5, what=...)` wrapper around the
kernel call (back off on "out of memory"/"hip" errors) is **recommended but not
required** (see `qk_norm_rope_quant_kernel`).

---

## 5. `kernel.py` — the FlyDSL TARGET (when present)

- **Self-contained, inline FlyDSL.** `import flydsl` directly; **inline the device
  kernel** (and its on-device shims) rather than importing an aiter/FlyDSL-repo
  *device* kernel. The only allowed `aiter` touches are **host-side data prep**
  (quant, e8m0 shuffle, `moe_sorting`, weight preshuffle) and lazy
  hardware-info queries — never a device-kernel import.
- Expose the function(s) named in `target_kernel_functions`: a public launcher
  (`flydsl_<op>(...)`), a module builder (`build_<op>_module(...)`) invoked by
  `compile_command`, and any inner compile entry.
- Same SPDX/header + clean-comment rules as §2.6. Upstream technical comments
  carried in with the inlined source are fine.
- Apple-to-apple: the inlined kernel must implement the SAME numerics the
  reference emulates, so the tight gate (§4) holds.

---

## 6. RULE — when a task ships `kernel.py` vs leaves it to GEAK

> **A task includes a FlyDSL `kernel.py` IFF aiter (or the FlyDSL repo) already
> has a clean, standalone FlyDSL device kernel that maps 1:1 to the op** (so it
> can be inlined and validated apple-to-apple against `model.py`).

- **YES — aiter has a matching standalone FlyDSL kernel** → full convention:
  inline that kernel as `kernel.py`, `compile_command` builds it,
  `target_kernel_functions` lists its entrypoints, and the harness validates
  kernel-vs-reference at the tight gate. *(8 of the 48 tasks are YES and ship an
  inlined `kernel.py`: `hgemm_kernel`, `gemm_a8w8_bpreshuffle_kernel`,
  `moe_kernel`, `moe_a8w4_kernel`, `moe_swiglu_kernel`, `moe_sorting_kernel`,
  `jagged_dense_bmm_kernel`, `qk_norm_rope_quant_kernel`.)* The other 40 are
  model-only (NO branch below): the warmup/elementwise/norm/rope/quant/routing
  ops where aiter has no clean standalone FlyDSL device kernel.

- **NO — no clean standalone FlyDSL kernel exists** (or only a fusion-specialized
  one with the wrong I/O layout) → ship **`model.py` + `config.yaml` +
  `test_kernel_harness.py` only, no `kernel.py`**. FlyDSL is GEAK's target. The
  harness runs the `model.py` reference and is wired to load `kernel.py` once GEAK
  produces it (the loader already handles a missing `kernel.py` by reporting a
  load failure → exit 1, which is the correct "not yet authored" state). Use the
  model-only `config.yaml` variant (§3).

Do **not** lazily fork an unrelated kernel or wrap a fusion-specialized kernel
behind a fake layout just to have a `kernel.py`; that produces a misleading
baseline. A genuine standalone kernel or nothing.

---

## 7. Batch A reconnaissance (gates the convention choice per op)

Checked `aiter/aiter/ops/flydsl/kernels/` @ aiter commit `e77319304` for a clean
standalone FlyDSL kernel per Batch A op, and located each op's (b)-faithful
op_test torch reference + real-shape source.

| Op | aiter FlyDSL standalone kernel? | → convention | op_test torch ref | AMD semantics / shapes |
|---|---|---|---|---|
| `silu_and_mul` | **NO** (only `kernels/silu_and_mul_fq.py` — a MoE-stage1 fused act+mul+**quant** + sorted-scale kernel; quant_mode="none"/act="silu" exists but I/O is MoE-shaped, not a standalone `silu_and_mul(x[...,2d])→y[...,d]`) | model-only (GEAK target) | `op_tests/test_activation.py:torch_silu_and_mul(input, limit=0.0)` | bf16; gate=x[...,:d], y=x[...,d:]; `F.silu(gate)*y`; `limit>0` ⇒ GPT-OSS clamp. Shapes: `--m × --n` sweep (`aiter.silu_and_mul`) |
| `swiglu_and_mul` | **NO** (only `kernels/swiglu_and_mul.py` — interleaved `(N0,2,NLane)` cktile a16w4-preshuffle layout, MoE-stage post-proc; not standalone) | model-only (GEAK target) | `op_tests/test_activation.py:torch_silu_and_mul(..., limit>0)` (clamped swiglu path) | bf16 clamped swiglu; shapes `--m × --n` (`aiter.swiglu_and_mul`) |
| `gelu_and_mul` | **NO** | model-only (GEAK target) | `op_tests/test_activation.py:torch_gelu_ref` | bf16 gelu·mul (`aiter.gelu_and_mul` / `gelu_tanh_and_mul`); shapes `--m × --n` |
| `rmsnorm2d` | **NO** | model-only (GEAK target) | `op_tests/test_rmsnorm2d.py:run_torch(input, weight, eps, residual=None)` | bf16 in/out, fp32 reduce, `F.rms_norm`; shapes `dim=(m,n)`, `l_m × l_n` (`aiter.rms_norm`/`rmsnorm2d_fwd`) |
| `fused_add_rmsnorm` | **NO** | model-only (GEAK target) | `op_tests/test_rmsnorm2dFusedAddQuant.py:run_torch` (residual-add+rmsnorm); also `op_tests/test_rmsnorm2d.py:run_torch(..., residual=...)` / triton `normalization/test_fused_rmsnorm_add.py:run_torch` | bf16 residual add then rmsnorm (`aiter.rmsnorm2d_fwd_with_add`); shapes `(m,n)` |
| `layernorm2d` | **NO** | model-only (GEAK target) | `op_tests/test_layernorm2d.py:run_torch(input, weight, bias, eps, residual=None, x_bias=None)` | bf16 layernorm, optional fused add; shapes `(m,n)` (`aiter.layernorm2d_fwd`) |
| `rope_fwd` | **NO standalone** (only the already-DONE fused `qk_norm_rope_quant.py`) | model-only (GEAK target) | `op_tests/test_rope.py:ref_rope_sbhd_fwd` (+ `ref_rope_thd_fwd`, `ref_rope_2d_fwd`) | bf16 cached cos/sin rotary; neox/gptj `rotate_style`, `reuse_freqs_front_part`, `nope_first`; sbhd `(s,b,h,d)` shapes (`aiter.rope_fwd`/`rope_cached_fwd`) |

**Decision for Batch A: ALL 7 ops are "NO" → every Batch A task is model-only**
(`model.py` + `config.yaml` + `test_kernel_harness.py`, FlyDSL left as GEAK's
target). The two ops that *have* a FlyDSL file (`silu_and_mul`, `swiglu_and_mul`)
have only **fusion-specialized** kernels (quant + sorted/interleaved preshuffle
layouts) that do not map 1:1 to the standalone activation op, so per the §6 rule
they do **not** ship a `kernel.py`.

> The mxfp4/e8m0 emulation for Batch B (quant) tasks should be **copied from
> `moe_kernel/model.py`**, per `AITER_OPS_CATALOG.md`.
