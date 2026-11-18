---
name: xparity
description: >-
  Verifies numerical accuracy, bitwise ULP parity, whole-output and row-wise
  relative L2 for FP8/FP4 dequantization kernels, Float64 and Pinned-Float32
  oracle grounding, and multi-regime synthetic tensor correctness (Student-t
  heavy tails, scattered and per-channel outliers, cancellation pairs, discrete
  bounded indices, and attention masks) for JAX, Pallas, Triton, XLA, PyTorch,
  and NumPy kernels using xparity_cli and the xparity Python library. Use when
  validating kernel optimizations, auditing reference baselines for silent TPU
  MXU BF16 or GPU TF32 truncation, diagnosing numerical divergence or false
  greens/reds under the Three Questions Framework, or generating stress test
  suites. Don't use for profiling hardware execution time, roofline efficiency,
  or trace events (use xprof instead).
---

<!-- disableFinding(LINE_OVER_80) -->

# Xparity: Numerical Accuracy & Parity Verification Skill

`xparity` is the standalone numerical correctness and hardware-grounded parity
verification engine for autonomous kernel engineering. Always verify numerical
equivalence and oracle grounding before claiming any kernel optimization
speedup.

## Quick Start

### 1. CLI Execution (`xparity`)

Run the `xparity` console script installed with the `xprof` package
(`pip install -U xprof`), or build it from source with
`bazel run //plugin/xprof/xparity:xparity_cli --`:

```bash
# Verify numerical parity + Float64 Oracle Audit between two callables
xparity verify \
  --kernel_ref="my_pkg.kernels:ref_fn" \
  --kernel_candidate="my_pkg.kernels:cand_fn" \
  --shapes="[(32, 2048)]" \
  --dtype_str="bfloat16" \
  --tier="presubmit" \
  --max_allowed_ulp=2 \
  --kernel_oracle="auto"

# Verify exact bit-pattern equality across all regimes (distinguishes -0.0/+0.0)
xparity verify \
  --kernel_ref="my_pkg.kernels:ref_fn" \
  --kernel_candidate="my_pkg.kernels:cand_fn" \
  --shapes="[(32, 2048)]" \
  --dtype_str="bfloat16" \
  --contract="bitwise"

# Gate a deliberately lossy kernel (FP8/FP4 dequantization) on whole-output
# and per-row relative L2 instead of ULP; --rel_l2_tol is required
xparity verify \
  --kernel_ref="my_pkg.kernels:ref_fn" \
  --kernel_candidate="my_pkg.kernels:fp8_dequant_fn" \
  --shapes="[(256, 4096)]" \
  --dtype_str="float32" \
  --contract="row_rel_l2" \
  --rel_l2_tol=2e-2 \
  --row_factor=5 \
  --row_norm_floor=0.1 \
  --verdict_only

# Generate and persist a multi-regime .npz stress test suite
xparity generate_suite \
  --shapes="[(16, 1024)]" \
  --output_path="/tmp/stress_suite.npz" \
  --dtype_str="float32" \
  --tier="presubmit"

# Probe whether a reference callable is precision-pinned or pin-inert on TPU/GPU
xparity probe_precision \
  --kernel_fn="my_pkg.kernels:ref_fn" \
  --shapes="[(128, 128)]" \
  --dtype_str="float32" \
  --device_kind="tpu"

# Replay a saved failing batch (.npz from --dump_failures_to) to verify a fix
# (pass --verdict_only on verify or replay to emit a compact verdict summary
# with only failing batches instead of the full per-batch payload)
xparity replay \
  --dump_path="/tmp/xparity_failures/xparity_failure_000_student_t_batch_0.npz" \
  --kernel_ref="my_pkg.kernels:ref_fn" \
  --kernel_candidate="my_pkg.kernels:cand_fn" \
  --verdict_only
```

### 2. Python Library API (`xprof.xparity`)

When validating inline callables inside a test or benchmark script, import
directly from `xprof.xparity`:

```python
from xprof.xparity import CONTRACT_BITWISE
from xprof.xparity import CONTRACT_ROW_REL_L2
from xprof.xparity import chunk_callable
from xprof.xparity import compare_bitwise
from xprof.xparity import compare_row_rel_l2
from xprof.xparity import generate_attention_suite
from xprof.xparity import make_fwd_bwd
from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import read_suite_metadata
from xprof.xparity import replay_failure_dump
from xprof.xparity import validate_kernels

report = validate_kernels(
    kernel_ref=reference_fn,
    kernel_candidate=candidate_fn,
    shapes=[(32, 2048)],
    dtype_str="bfloat16",
    tier="presubmit",
    max_allowed_ulp=2,
    kernel_oracle="auto",
    contract="ulp",  # or CONTRACT_BITWISE for exact bit-pattern equality
    dump_failures_to="/tmp/xparity_failures",
    # Optional per-leaf contract overrides for tuple/dict/pytree outputs:
    # contract_by_leaf={"['out']": "bitwise", "['vjp'][0]": "ulp"},
)

# Re-run a saved failing batch directly after editing candidate_fn:
if report.failure_dumps:
  replay_report = replay_failure_dump(
      report.failure_dumps[0], reference_fn, candidate_fn
  )

# Deliberately lossy kernel (FP8/FP4 dequantization): gate on relative L2.
# rel_l2_tol is required; row_factor=5.0 and row_norm_floor=0.1 by default.
row_report = validate_kernels(
    kernel_ref=reference_fn,
    kernel_candidate=fp8_dequant_fn,
    shapes=[(256, 4096)],
    dtype_str="float32",
    contract=CONTRACT_ROW_REL_L2,
    rel_l2_tol=2e-2,
)
# Or compare two outputs directly (NumPy, JAX or PyTorch, on any device):
row_check = compare_row_rel_l2(candidate_out, reference_out, tol=2e-2)
if not row_check.passed:
  print(row_check.reason, row_check.worst_row_index, row_check.worst_row_rel_l2)

# Validate both forward output and backward VJP gradients leaf-by-leaf:
fwd_bwd_report = validate_kernels(
    kernel_ref=make_fwd_bwd(reference_fn),
    kernel_candidate=make_fwd_bwd(candidate_fn),
    shapes=[(32, 2048)],
    dtype_str="bfloat16",
)
```

--------------------------------------------------------------------------------

## Core Mandate: The Three Questions Framework & Inverted Verdict Reading

Never read `is_numerically_equivalent` in isolation. Two kernels that both
truncate to `bfloat16` on the TPU MXU agree at `0 ULP`
(`is_numerically_equivalent: true`) while sitting `>15,000x` away from
mathematical ground truth (a **False Green**). Conversely, a parallel tree or
Split-K reduction reorders non-associative floating-point additions and exceeds
`2 ULP` relative to sequential accumulation while sitting *closer* to the true
Float64 oracle (a **False Red**).

### Mandatory Verdict-Reading Order (Inverted Order)

1.  **Check `run_config` & `coverage` Provenance First**: Confirm `tier`
    (`fast_agent` vs `presubmit`), `contract` (`"ulp"`, `"bitwise"` or `"row_rel_l2"`),
    `dtype_str`, `device_kind`, `total_batches_count`, and `coverage`
    (`selection`, `regimes_run`, `regimes_not_run`). By default, `presubmit` and
    `deep_fuzzing` gate on every regime in the suite (`selection:
    "full_suite"`), whereas `fast_agent` runs `normal` first with triage on
    failure (`selection: "normal_first"`). Never quote a `fast_agent` ULP figure
    or a run with non-empty `regimes_not_run` as a final `presubmit`
    certification.
2.  **Check `tolerance_audit` & `tolerance_headroom` Second**: Verify
    `configured_max_ulp` against `recommended_contract_ulp` (`2 ULP` for
    `float32`/`bfloat16`/`float16`, `1 ULP` for `fp8`, `0 ULP` for discrete
    `int*`/`bool`) and the immutable `hard_safety_ceiling` (`8 ULP` for
    `bfloat16`/`float16`, `4 ULP` for `float32`). Any attempt to set
    `max_allowed_ulp` above `hard_safety_ceiling` raises a `ValueError`. When a
    downstream test suite has existing `atol`/`rtol` budgets, pass `atol` and
    `rtol` (`--atol`, `--rtol`) to populate report-only `tolerance_headroom`
    (`max_ratio`, `exceeded`, `worst_batch`, `oracle_max_ratio`,
    `oracle_exceeded`, and per-batch `tolerance_ratio` /
    `oracle_tolerance_ratio`, where $\le 1.0$ means within `atol + rtol * |ref|`)
    without altering `is_numerically_equivalent`.
3.  **Check `oracle_audit` Third (Questions Q2 & Q3)**:
    -   **Q2 (Reference Correctness — `reference_is_lossy`)**: Is
        `reference_max_ulp_from_oracle <= recommended_contract_ulp`? If
        `reference_is_lossy: true` (or `reference_pin_inert: true`), the
        reference baseline itself is unpinned or truncating (`Precision.DEFAULT`
        on TPU MXU or TF32 on GPU). Pairwise agreement with a lossy reference
        does NOT establish correctness (`correctness_basis` must be
        `"AGREEMENT_AND_ORACLE"` with `oracle_precision_verified: true`).
    -   **Q3 (Accuracy Drift — `candidate_max_ulp_from_oracle` vs
        `reference_max_ulp_from_oracle`)**: Did the candidate improve upon or
        degrade the reference relative to the Float64 oracle? For reassociating
        optimizations (e.g., tree reduction, Split-K), do **NOT** use pairwise
        `verify_numerical_parity` (`Q1`) as an automated merge gate; evaluate
        `Q3` against the Float64 oracle across the **full procedural suite**
        (`regimes="all"`).
4.  **Check `is_numerically_equivalent` & `batch_results` Last (Question Q1 —
    Behavior Alteration)**:
    -   Inspect `worst_offender` (`max_ulp_index`, `ref_value`, `cand_value`,
        `abs_diff`, `rel_diff`, `mismatch_count`), `bitwise` (`equal`,
        `diff_count`, `diff_ratio`, `first_diff_index`, `reference_bits`,
        `candidate_bits`), and non-finite telemetry (`nan_count`, `inf_count`,
        `first_non_finite_index`, `finite_max_ulp`) to pinpoint localized
        boundary, sign-of-zero, causal-diagonal, or gather-index defects. When
        requiring exact bit identity across layout or refactor changes, pass
        `contract="bitwise"` (do not gate on `overall_max_ulp == 0`, which maps
        `-0.0` and `+0.0` to the same index and fails on identical `NaN`s).
    -   For multi-output (pytree) kernels or `make_fwd_bwd` backward-pass
        checks, inspect `leaf_path` (the first failing leaf or worst-ULP leaf,
        e.g. `"[0]"`, `"['lse']"`, or `"['vjp'][0]"`) and per-leaf entries in
        `leaf_results`. Every leaf must pass for the batch to pass; use
        `contract_by_leaf` (`--contract_by_leaf`) to set per-leaf contracts.
    -   Under `contract="row_rel_l2"`, the verdict comes from each batch's (or
        leaf's) `row_rel_l2`: `rel_l2` (whole output), `worst_row_rel_l2` and
        `worst_row_index` (rows run over the last dimension), `rows_over_bound`,
        and `reason` (`whole_output_rel_l2`, `row_rel_l2`,
        `non_finite_candidate`, `non_finite_reference`, `shape_mismatch` or
        `dtype_mismatch`). The report's `rel_l2_summary` gives `max_rel_l2`,
        `max_row_rel_l2`, `worst_row_batch`, `worst_row_leaf`,
        `worst_row_index` and `failed_checks`; `--verdict_only` repeats these
        per failing batch and under a top-level `rel_l2`. ULP fields are still
        reported but do not gate.

**Reporting the verdict.** State the result with the report's own field names
so it can be checked against the JSON: `passed: true|false` (from
`is_numerically_equivalent`), `max_ulp_distance`, and the worst relative error
(`worst_offender.rel_diff`, or `rel_l2_summary.max_rel_l2` and
`max_row_rel_l2` under `row_rel_l2`), plus the failing batch count.

--------------------------------------------------------------------------------

## When to Prefer `row_rel_l2` Over ULP

The ULP contract asks whether every element is within a few representable steps
of the reference. Keep it (the default) when the candidate should reproduce the
reference up to rounding: fusion, layout changes, same-precision rewrites. Use
`contract="row_rel_l2"` when the candidate's arithmetic is deliberately lossier
than the reference's and correctness is defined in aggregate:

-   **FP8 / FP4 dequantization** (`float8_e4m3fn`, `float8_e5m2`,
    `float4_e2m1fn` weights or activations, e.g. W4A16 or W8A8 GEMM and MoE
    kernels) checked against a `bfloat16` / `float32` reference. A row-scaled
    `float8_e4m3fn` value keeps 3 mantissa bits, so single elements can sit up
    to about $2^{19}$ `float32` ULP from the reference while the relative L2
    error is small. No `max_allowed_ulp` under `hard_safety_ceiling` accepts that, and
    raising the ULP budget would also hide real defects.
-   **Kernels whose upstream acceptance test is a relative-L2 norm**, so the
    xparity gate matches the threshold the kernel is specified against.

The check passes when both hold, with $c$ the candidate, $r$ the reference and
rows $i$ running over the last dimension:

$$\frac{\lVert c - r \rVert_2}{\lVert r \rVert_2} \le \texttt{rel\_l2\_tol}
\quad\text{and}\quad
\max_i \frac{\lVert c_i - r_i \rVert_2}{\max(\lVert r_i \rVert_2,\;
\texttt{row\_norm\_floor} \cdot \overline{\lVert r_j \rVert_2})} \le
\texttt{row\_factor} \cdot \texttt{rel\_l2\_tol}$$

-   The row bound catches one wrong row (a misrouted token, a bad expert or
    tile) that the whole-output norm averages away; the floor keeps rows that
    are legitimately near zero from inflating their ratio.
-   NaN or Inf in either output, a shape mismatch, or an output dtype that
    differs from the reference's fails the check.
-   There is no default `rel_l2_tol`: take it from the kernel's own acceptance
    spec and never loosen it to make a candidate pass. Keep `row_factor=5` and
    `row_norm_floor=0.1` unless the spec says otherwise.
-   Pairwise relative L2 against a lossy reference still proves nothing about
    ground truth; read `oracle_audit` (Q2 and Q3) as for the ULP contract.
-   Inputs are generated as NumPy arrays in `dtype_str`, and there is no
    `float4_e2m1fn` generator profile: pass the high-precision input dtype and
    quantize inside the candidate. `float4_e2m1fn` outputs are accepted under
    every contract (the ULP contract for them is 0 ULP), but for dequantization
    kernels gate the dequantized `bfloat16` / `float32` output with
    `row_rel_l2`. PyTorch outputs, including CUDA
    tensors, are converted to NumPy automatically, but PyTorch kernels must
    convert their NumPy inputs themselves.

--------------------------------------------------------------------------------

## Multi-Regime Stress Generators (`numerical_generator`)

Benign Gaussian (`randn`) inputs stay within `[-4, 4]` and fail to trigger
`exp(x)` overflow (`> 88.72` in FP32), catastrophic cancellation, or
out-of-bounds gather routing. Always use `numerical_generator` regimes matched
to the kernel's failure modes:

-   **Continuous Float Regimes (`generate_test_suite`)**:
    -   `normal` (`generate_normal_tensor`): Benign unit-Gaussian baseline
        ($\sigma = 1$).
    -   `scale_sweep` (`generate_normal_tensor(..., scale=sigma)` with
        `SCALE_SWEEP_SIGMAS = (0.02, 3.0, 10.0)`): Sub-unit and super-unit
        Gaussian magnitudes to expose scale-dependent softmax/attention and
        normalization branches that unit-variance inputs miss.
    -   `student_t` (`generate_student_t_tensor`): Heavy-tailed power-law draws
        ($\nu \in [2.5, 4.0]$) bounded at $0.95 \times \text{max\_finite}$.
    -   `outliers` (`generate_outlier_tensor`): Scattered $50\times$ activation
        spikes.
    -   `per_channel_outliers` (`generate_per_channel_outlier_tensor`):
        Channel-aligned LLM activation spikes (SmoothQuant/AWQ regime where 1–2%
        of channels are amplified across all tokens).
    -   `cancellation` (`generate_cancellation_tensor`): Alternating $+M, -M +
        \epsilon$ pairs with dynamic $\ge 2\text{ ULP}$ floor to expose
        accumulator truncation (`bf16` vs `fp32`).
    -   `boundary` (`generate_boundary_probe_tensor`): `min_normal`,
        `min_subnormal`, `0.0`, and $\pm 10^4$ aligned to 128-byte TPU VMEM tile
        strides.
-   **Attention Suite Generator (`generate_attention_suite`)**:
    -   Generates `(q, k, v, mask_or_segment_ids)` batches for MHA/GQA/MQA
        (`q_shape=(B, H_q, S_q, D)`, `num_kv_heads=H_kv`) across five
        attention-specific regimes: `attention_normal` (unit $Q, K, V$ with
        all-True mask), `attention_large_logits` ($\sigma = 8.0$ on $Q, K$ to
        stress online-softmax rescaling before `exp` overflow),
        `attention_causal` (lower-triangular mask supporting $S_q \ne S_{kv}$),
        `attention_padding` (variable sequence-length key padding with a fully
        unmasked query row), and `attention_segments` (packed-sequence segment
        equality mask `seg_q == seg_kv`, or raw `(q, k, v, seg_q, seg_kv)` when
        `pass_segment_ids=True`).
-   **Discrete & Mask Regimes (`max_allowed_ulp = 0`)**:
    -   `generate_index_tensor(shape, upper_bound, lower_bound=0,
        include_boundaries=True)`: Bounded indices in `[lower_bound,
        upper_bound - 1]` with deterministic pinning of `0` and `upper_bound -
        1` (for MoE expert routing and gather/scatter).
    -   `generate_segment_ids_tensor(shape, num_segments, is_sorted=True)`:
        Monotonically non-decreasing segment IDs for ragged/segmented
        reductions.
    -   `generate_mask_tensor(shape, mask_type="causal" | "bernoulli" |
        "padding")`: 2D/3D/4D broadcastable boolean and integer masks.

--------------------------------------------------------------------------------

## Reference Documentation

-   **[references/numerical_correctness.md](references/numerical_correctness.md)**:
    Complete mathematical specification, ULP sign-magnitude integer mapping,
    Dual Gating table, Certified Pinned Float32 ($100\times$ Precision Margin
    Rule), OOM chunking (`chunk_callable`), and regime dispatch rules.
