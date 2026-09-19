---
name: xparity
description: >-
  Verifies numerical accuracy, bitwise ULP parity, Float64 and Pinned-Float32
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

Run the `xparity` console script installed via `pip install xprof` (or `bazel run //plugin/xprof/xparity:xparity_cli --`):

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
```

### 2. Python Library API (`xprof.xparity`)

When validating inline callables inside a test or benchmark script, import
directly from `xprof.xparity`:

```python
from xprof.xparity import chunk_callable
from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import validate_kernels

report = validate_kernels(
    kernel_ref=reference_fn,
    kernel_candidate=candidate_fn,
    shapes=[(32, 2048)],
    dtype_str="bfloat16",
    tier="presubmit",
    max_allowed_ulp=2,
    kernel_oracle="auto",
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

1.  **Check `run_config` Provenance First**: Confirm `tier` (`fast_agent` vs
    `presubmit`), `dtype_str`, `device_kind`, and `total_batches_count`. Never
    quote a `fast_agent` ULP figure as a final `presubmit` certification.
2.  **Check `tolerance_audit` Second**: Verify `configured_max_ulp` against
    `recommended_contract_ulp` (`2 ULP` for `float32`/`bfloat16`/`float16`, `1
    ULP` for `fp8`, `0 ULP` for discrete `int*`/`bool`) and the immutable
    `hard_safety_ceiling` (`8 ULP` for `bfloat16`/`float16`, `4 ULP` for
    `float32`). Any attempt to set `max_allowed_ulp` above `hard_safety_ceiling`
    raises a `ValueError`.
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
        `abs_diff`, `rel_diff`, `mismatch_count`) and non-finite telemetry
        (`nan_count`, `inf_count`, `first_non_finite_index`, `finite_max_ulp`)
        to pinpoint localized boundary, causal-diagonal, or gather-index
        defects.

--------------------------------------------------------------------------------

## Multi-Regime Stress Generators (`numerical_generator`)

Benign Gaussian (`randn`) inputs stay within `[-4, 4]` and fail to trigger
`exp(x)` overflow (`> 88.72` in FP32), catastrophic cancellation, or
out-of-bounds gather routing. Always use `numerical_generator` regimes matched
to the kernel's failure modes:

-   **Continuous Float Regimes (`generate_test_suite`)**:
    -   `normal`: Benign Gaussian baseline.
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
