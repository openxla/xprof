# Numerical Correctness Verification in XProf

This guide provides the complete operational methodology and reference for
verifying numerical equivalence between baseline reference implementations and
optimized compute kernels (TPU, GPU, CPU, Pallas, Triton, C++, CUDA) across
compiler transformations, lowerings, and autotuning.

--------------------------------------------------------------------------------

## 1. Overview & Situational Use

The numerical capability serves two well-defined, unassailable roles:

1.  **Reference Grounding Check**: *"Is your baseline what you think it is?"*
    Detects silently lossy references using the Float64 Oracle
    (`kernel_oracle`). Accelerator execution often silently truncates references
    (e.g. TPU MXU bf16 passes under default precision, GPU TF32).
2.  **Refactor Equivalence Check**: *"Did this code cleanup, rename, or layout
    reshape alter bits?"* Evaluates exact or tight ULP equivalence without an
    oracle.

> [!WARNING] **NOT an Automated Merge Gate for Optimizations**: Do **NOT** use
> `verify_numerical_parity` as an automated merge gate to block kernel
> optimizations that alter reduction associativity ($a + (b + c) \ne (a + b) +
> c$, such as tree reductions vs sequential accumulation). Fused reductions and
> reassociation naturally cause large ULP deviations (e.g. $8{,}388{,}608\text{
> ULP}$) while remaining mathematically sound or even closer to true ground
> truth than the baseline.

### Situational Verification Matrix

| Workload / Domain     | Target Operator & Use   | Verification Approach      |
:                       : Case                    :                            :
| :-------------------- | :---------------------- | :------------------------- |
| **Continuous          | MatMul, FlashAttention, | Default normal regime with |
: Floating-Point**      : RMSNorm, Activations    : automated triage fallback  :
:                       :                         : (Student-t, outliers,      :
:                       :                         : cancellation, boundary).   :
| **Discrete / Token    | MoE Expert Routing,     | Bounded index generation   |
: Routing**             : Embedding Lookups,      : with deterministic         :
:                       : Gather                  : boundary extreme pinning   :
:                       :                         : ($0, N-1$).                :
| **Ragged Reductions** | `segmented_sum`, ragged | Monotonic non-decreasing   |
:                       : token batch reductions  : segment ID partitioning    :
:                       :                         : along reduction axis.      :
| **Attention           | Causal, Sparse, and     | Structured boolean mask    |
: Topologies**          : Sequence Padding Masks  : generation with 4D $(B, H, :
:                       :                         : S, S)$ multi-head          :
:                       :                         : attention broadcasting.    :
| **Quantized           | INT8/INT4 quantization, | Integer multi-regime       |
: Arithmetic**          : integer accumulators    : suites testing extreme     :
:                       :                         : boundaries (`min_val`,     :
:                       :                         : `max_val`, $0, 1, -1$).    :

--------------------------------------------------------------------------------

## 2. Quick Start Workflows

### Workflow A: CLI Tool (`xprof verify_numerical_parity`)

⚠️ **MANDATORY**: When asked to verify reference grounding or refactor
equivalence, you **MUST** run the `xprof verify_numerical_parity` CLI command
(or `xprof_cli verify_numerical_parity`) rather than writing a custom inline
script.

```bash
# Verify parity between two Python callables using the fast_agent tier
# with automatic Float64 Oracle audit enabled
xprof verify_numerical_parity \
  --kernel_ref="my_module.reference_fn" \
  --kernel_candidate="my_module.optimized_fn" \
  --kernel_oracle="auto" \
  --shapes="[(16, 1024), (1024, 1024)]" \
  --dtype_str="bfloat16" \
  --tier="fast_agent"
```

#### JSON Output Schema

```json
{
  "is_numerically_equivalent": true,
  "overall_max_ulp": 1,
  "failed_batches_count": 0,
  "total_batches_count": 1,
  "summary_message": "PASSED: Kernels are numerically equivalent across 1 batches (Max ULP: 1, Configured Limit: 2, Recommended: <= 2).",
  "correctness_basis": "AGREEMENT_AND_ORACLE",
  "run_config": {
    "tier": "fast_agent",
    "seed": 42,
    "dtype_str": "bfloat16",
    "device_kind": "tpu_v6e",
    "backend": "tpu",
    "total_batches_count": 1
  },
  "tolerance_audit": {
    "recommended_contract_ulp": 2,
    "configured_max_ulp": 2,
    "hard_safety_ceiling": 8,
    "is_relaxed_override": false,
    "caution_banner": null
  },
  "oracle_audit": {
    "oracle_executed_in_float64": true,
    "oracle_output_dtype": "float64",
    "reference_max_ulp_from_oracle": 1,
    "reference_p99_9_ulp_from_oracle": 1.0,
    "candidate_max_ulp_from_oracle": 1,
    "candidate_p99_9_ulp_from_oracle": 1.0,
    "reference_is_lossy": false,
    "reference_max_abs_from_oracle": 0.0078125,
    "candidate_max_abs_from_oracle": 0.0078125,
    "oracle_banner": null
  },
  "ulp_context": {
    "bit_identical": false,
    "p50": 0.0,
    "p99_9": 1.0,
    "max_ulp": 1,
    "reliable": true,
    "note": null
  },
  "batch_results": [
    {
      "batch_name": "normal_batch_0",
      "regime": "normal",
      "max_ulp_distance": 1,
      "p99_9_ulp_distance": 1.0,
      "mean_ulp_distance": 0.04,
      "reference_ulp_from_oracle": 1,
      "candidate_ulp_from_oracle": 1,
      "ulp_histogram": {
        "<=1_ulp": 16384,
        "<=2_ulp": 16384,
        ">2_ulp": 0
      },
      "has_nan_or_inf": false,
      "passed": true,
      "allclose_passed": true,
      "ulp_context": {
        "bit_identical": false,
        "p50": 0.0,
        "p99_9": 1.0,
        "max_ulp": 1,
        "reliable": true,
        "note": null
      }
    }
  ]
}
```

### Workflow B: Python API (`validate_kernels`)

For programmatic integration within Python test harnesses or optimization loops:

```python
from xprof.cli.internal import numerical_validator

# 1. Define Reference and Candidate Kernels
# CRITICAL: Always pin reference precision to HIGHEST. On TPU, unpinned jnp.dot
# truncates f32 inputs to bf16 for the MXU (a 15,335x precision gap).
def ref_kernel(a, b):
  return jnp.dot(a, b, precision=jax.lax.Precision.HIGHEST)

def candidate_kernel(a, b):
  return custom_refactored_matmul(a, b)

# 2. Validate Parity with Float64 Oracle Audit
report = numerical_validator.validate_kernels(
    kernel_ref=ref_kernel,
    kernel_candidate=candidate_kernel,
    kernel_oracle="auto",  # or custom float64 callable
    shapes=[(128, 64), (64, 128)],
    dtype_str="bfloat16",
    tier="presubmit",
    max_allowed_ulp=2,
    p99_9_allowed_ulp=1,
)

if not report.is_numerically_equivalent:
  print(f"Validation FAILED: {report.summary_message}")
  for batch in report.batch_results:
    if not batch.passed:
      print(f"  Batch {batch.batch_name}: Max ULP={batch.max_ulp_distance}")
```

### Operational Testing Tiers

Tier               | Total Tensors ($m$)      | Composition                                                        | Latency                       | Recommended Use
:----------------- | :----------------------: | :----------------------------------------------------------------- | :---------------------------: | :--------------
**`fast_agent`**   | **$m = 5$**              | 2 Student-t + 1 Outlier ($50\times$) + 1 Cancellation + 1 Boundary | $\sim 1\text{--}2\text{ s}$   | Interactive pair-programming iteration by agent
**`presubmit`**    | **$m = 12\text{--}15$**  | 6-8 Student-t + 3-4 Outliers + 3 Boundary probes                   | $\sim 5\text{--}8\text{ s}$   | Automated presubmit before submitting CL
**`deep_fuzzing`** | **$m = 50\text{--}100$** | 30 Student-t + 15 Outliers + 5 Boundary grids                      | $\sim 30\text{--}60\text{ s}$ | Compiler pass / custom kernel release qualification

--------------------------------------------------------------------------------

## 3. Float64 Oracle Audit & The Three Questions Framework

### 3.1 The Hardware Precision Trap & Precision Pinning Prerequisite

Numerical equivalence between a candidate kernel and a reference kernel does not
guarantee mathematical correctness if the reference itself is truncated by
hardware defaults:

*   **TPU Matrix Multiply Units (MXUs)**: Under `jax.lax.Precision.DEFAULT`,
    `jnp.dot` truncates FP32 inputs to BF16 for matrix multiplications.
*   **Ampere+ GPUs**: TF32 (TensorFloat-32) execution truncates FP32 mantissas
    from 23 bits to 10 bits unless disabled.
*   **CPU**: Pure IEEE-754 FP32 execution.

Consequently, identical reference code that passes on CPU can silently produce a
false green on TPU/GPU when compared against a low-precision candidate.

**Mandatory Precision Pinning**: When authoring reference implementations,
always pin internal operations to maximum precision:

```python
# JAX: Pin dot products and convolutions to HIGHEST precision
result = jnp.dot(a, b, precision=jax.lax.Precision.HIGHEST)

# PyTorch: Disable TF32 mantissa truncation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### 3.2 Operating the Float64 Oracle (`kernel_oracle`)

To determine whether the reference itself is mathematically grounded,
`validate_kernels` and `verify_numerical_parity` accept an optional
`kernel_oracle` parameter:

1.  **Automatic Promotion (`kernel_oracle="auto"`)**: Promotes all
    floating-point input tensors to `float64`, executes `kernel_ref`, and
    measures reference vs. oracle error. Non-floating arguments (discrete
    indices, segment IDs, boolean masks) pass through untouched.
2.  **Explicit Callable (`kernel_oracle=fn`)**: Supplies an analytical ground
    truth implementation written in native `float64` (e.g. using
    `scipy.special`, `mpmath`, or high-precision math).

#### Mechanics and Blind Spots of `kernel_oracle="auto"`

`kernel_oracle="auto"` is a lightweight, zero-code mechanism that catches
numerical loss in *precision-following* references — implementations where
intermediate operations inherit their precision from their input arguments (such
as `jnp.dot(a, b)` under `Precision.DEFAULT`).

However, `kernel_oracle="auto"` sees **only** the loss that the input dtypes
control:

*   **Hardcoded Internal Casts**: If `kernel_ref` explicitly downcasts tensors
    inside its body (e.g. `x.astype(jnp.bfloat16)`), promoting inputs to
    `float64` does not lift the internal cast.
*   **Hardcoded Constant Literals**: Literals instantiated with explicit 32-bit
    or 16-bit dtypes (e.g. `jnp.array(1.0, dtype=jnp.float32)`) remain fixed.
*   **Fixed Precision Hardware Ops**: Operations whose hardware execution
    precision does not scale with `float64` inputs will not be elevated.

When a reference has hardcoded internal casts, the output array fails to execute
in `float64`. In this event, `numerical_validator` refuses to emit a false clean
bill of health: `oracle_audit.oracle_executed_in_float64` evaluates to `False`
and an explicit caution banner is emitted (`"⚠️ ORACLE DID NOT EXECUTE IN
FLOAT64"`). To certify such kernels, an **explicit float64 callable** is
required.

Property                         | Automatic Promotion (`kernel_oracle="auto"`)                | Explicit Float64 Callable (`kernel_oracle=fn`)
:------------------------------- | :---------------------------------------------------------- | :---------------------------------------------
**Input Dtype Promotion**        | Promotes all floating inputs to `float64`                   | Caller supplies native `float64` callable
**Precision-Following Ops**      | ✅ Detected (re-executes at `float64`)                       | ✅ Detected
**Hardcoded Internal Casts**     | ❌ Blind (guarded by float64 output check)                   | ✅ Detected
**Hardcoded Constant Literals**  | ❌ Blind                                                     | ✅ Detected
**JAX `jax_enable_x64` Setting** | **Mandatory** (`jax.config.update("jax_enable_x64", True)`) | **Mandatory** for JAX oracles (or use host NumPy/SciPy)
**Effort & Friction**            | Zero code (drop-in string `"auto"`)                         | Requires authoring high-precision callable
**Recommended Use**              | Quick hardware check                                        | Qualification of compiler passes & critical math kernels

> [!WARNING] **JAX Disables 64-Bit Precision by Default**: JAX defaults to
> 32-bit floating-point arithmetic. If `jax.config.update("jax_enable_x64",
> True)` is not executed at runtime initialization, JAX silently downcasts
> `float64` inputs back to `float32`. When this occurs,
> `oracle_audit.oracle_executed_in_float64` evaluates to `False` and the oracle
> metrics cannot serve as a mathematical correctness bound.

**CLI Usage**:

```bash
xprof verify_numerical_parity \
  --kernel_ref="my_module.reference_fn" \
  --kernel_candidate="my_module.optimized_fn" \
  --kernel_oracle="auto" \
  --shapes="[(16, 1024), (1024, 1024)]" \
  --dtype_str="bfloat16" \
  --tier="fast_agent"
```

**Zero Blast Radius**: The oracle audit is strictly report-only
(`oracle_audit`). It does not alter `is_numerically_equivalent` or block
existing pipelines, ensuring backwards compatibility.

### 3.3 The Three Questions Framework

A complete numerical investigation must answer three distinct questions:

1.  **Q1 (Behavior Alteration)**: Did the optimized candidate alter the
    reference implementation's behavior?
    *   *Metric*: Candidate vs. Reference ULP distance.
    *   *Gate*: `overall_max_ulp <= max_allowed_ulp` governs
        `is_numerically_equivalent`.
2.  **Q2 (Reference Correctness)**: Is the reference implementation itself
    numerically sound, or is it lossy?
    *   *Metric*: Reference vs. Oracle ULP distance.
    *   *Indicator*: `oracle_audit.reference_is_lossy` (`true` if ULP distance
        from float64 oracle exceeds the recommended contract ULP for the output
        dtype, e.g. > 1 ULP for float64/fp8, > 2 ULP for
        float32/bfloat16/float16, > 0 ULP for discrete/integers).
3.  **Q3 (Accuracy Drift)**: Did the candidate improve upon or degrade the
    reference relative to true mathematical ground truth?
    *   *Metric*: `candidate_max_ulp_from_oracle` vs.
        `reference_max_ulp_from_oracle`.

### 3.4 Inverted Verdict-Reading Order for Autonomous Agents

Autonomous agents must not rely solely on `is_numerically_equivalent: true`.
Follow this inverted hierarchy when parsing verification reports:

1.  **Step 1: Check Execution Context & Provenance (`run_config`)**: Inspect
    `tier`, `device_kind` / `backend`, `dtype_str`, and `total_batches_count`.
    Establish whether you are operating in `fast_agent` ($m=5$ smoke test) vs
    `presubmit` ($m=15$ gating), and whether execution ran on TPU, GPU, or CPU.
2.  **Step 2: Check Correctness Basis (`correctness_basis`)**: Verify whether
    the claim is `"AGREEMENT_AND_ORACLE"` or `"AGREEMENT_ONLY"`. If
    `"AGREEMENT_ONLY"`, the verdict certifies pairwise consistency
    ($f_{\text{ref}} \approx f_{\text{cand}}$) only, NOT mathematical ground
    truth. Never claim a kernel is mathematically correct if
    `correctness_basis == "AGREEMENT_ONLY"`.
3.  **Step 3: Verify Float64 Execution Integrity**: Check
    `oracle_audit.oracle_executed_in_float64`. If `False`, the runtime failed to
    run in float64 (e.g. JAX x64 disabled). In this case, oracle metrics cannot
    serve as a mathematical correctness bound.
4.  **Step 4: Check Reference Grounding & Degradation (Q2 & Q3)**:
    *   Inspect `oracle_audit.reference_is_lossy` and
        `oracle_audit.oracle_banner`. If `True`, the baseline kernel itself is
        corrupted by accelerator truncation (e.g. TPU MXU BF16 or GPU TF32). A 0
        ULP candidate match is a false green.
    *   Inspect Q3 Accuracy Drift: compare `candidate_max_ulp_from_oracle` with
        `reference_max_ulp_from_oracle`. If candidate distance exceeds reference
        distance, the candidate introduces excess error beyond baseline
        rounding.
5.  **Step 5: Check Safety Overrides (`tolerance_audit.is_relaxed_override`)**:
    Inspect `tolerance_audit.is_relaxed_override`. If `True`, verify whether
    tolerance inflation ($\tau > \text{RECOMMENDED}$) masked an underlying
    numerical regression.
6.  **Step 6: Check Parity Verdict (Q1)**: Inspect `is_numerically_equivalent`.
    If `False`, the candidate altered reference behavior beyond allowable
    contract tolerances.
7.  **Step 7: Analyze ULP Distribution & Cancellation Context**: Only now
    inspect `overall_max_ulp`. If `overall_max_ulp` is elevated while `p99_9`
    and the bulk histogram (`<=1_ulp`, `<=2_ulp`) are well within contract,
    check for near-zero cancellation before assuming failure.

### 3.5 Diagnostic Absolute Deviation vs. Authoritative ULP Gate

The oracle audit reports `reference_max_abs_from_oracle` and
`candidate_max_abs_from_oracle` ($\max |y - y_{\text{oracle}}|$):

*   **ULP is the Authoritative Gate**: ULP distance is scale-invariant and
    strictly bounded by IEEE-754 bit representations.
*   **Absolute Error is Diagnostic Only**: While valuable for human physical
    intuition, absolute error is dominated by the largest output magnitudes and
    says nothing about near-zero regions. Relative error metrics are tracked in
    the Phase 2 follow-up queue (F3) and will remain strictly report-only
    diagnostics.

### 3.6 What This Tool Cannot Tell You

Empirical validation on accelerator silicon demonstrates key boundaries where
isolated kernel parity testing reaches its limits. Agents and developers must
understand these boundaries:

1.  **Downstream Training / Convergence Impact**: ULP distances and gate ratios
    do not predict training stability or model degradation. In practice, a
    $47{,}180\times$ ULP gate violation caused by accumulator reordering had
    zero impact on downstream loss convergence (mixed-precision training
    naturally absorbs small variance), whereas a $3{,}450\times$ scale bug
    severely degraded attention outputs. Never infer end-to-end model harm from
    a large ULP distance alone.
2.  **bf16 Output Destroys Evidence (The FP32 Output Rule)**: When testing a
    kernel that outputs `bfloat16`, the coarse mantissa quantization (7 bits,
    $\epsilon \approx 7.8 \times 10^{-3}$) hides subtle numerical bugs (e.g.
    $0.1\%$ scale shifts or low-order accumulator drift). **Rule**: Whenever
    possible during kernel development and refactoring, widen kernel outputs to
    `float32` for verification so the numerical validator can observe mantissa
    dynamics before quantization occurs.
3.  **Identical Kernels Cannot Be Separated Without an Oracle**: If both the
    baseline and candidate share an identical flaw (e.g. both downcast an
    intermediate reduction to `bfloat16` on TPU MXUs), pairwise comparison
    reports $0\text{ ULP}$ across all input regimes. Pairwise parity only proves
    the two callables agree; only `kernel_oracle` can evaluate whether they are
    mathematically sound.
4.  **Reduction Reassociation Causes Large ULP Deviations**: Altering the order
    of floating-point reduction (such as replacing sequential summation with
    pairwise tree reduction or tiled block accumulation in FlashAttention) can
    produce deviations up to $8{,}388{,}608\text{ ULP}$ in `float32`. This is
    not a bug; floating-point addition is non-associative ($a + (b + c) \ne (a +
    b) + c$). In many cases, tree reduction is closer to the true Float64 oracle
    than sequential summation. Never use ULP parity as an automated merge gate
    for optimizations that reassociate reductions.

--------------------------------------------------------------------------------

## 4. Bitwise ULP Tolerance Contracts & Anti-Abuse Guardrails

To prevent autonomous agents or optimization scripts from artificially inflating
tolerances to mask numerical bugs (reward hacking), `numerical_validator`
enforces **Recommended ULP Contracts** and **Immutable Hard Safety Ceilings**:

| Dtype Category | Dtypes | Recommended Contract (`RECOMMENDED_CONTRACT_ULP`) | Immutable Hard Ceiling (`MAX_HARD_CEILING_ULP`) |
| :--- | :--- | :---: | :---: |
| **Discrete / Integer** | `bool`, `int4`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` | **`0 ULP`** (Exact Equality) | **`0 ULP`** (Immutable) |
| **Micro-Float** | `float4_e2m1fn` | **`0 ULP`** | **`0 ULP`** |
| **8-Bit Floats** | `float8_e4m3fn`, `float8_e5m2` | **`1 ULP`** | **`2 ULP`** |
| **Standard Floats** | `bfloat16`, `float16`, `float32` | **`2 ULP`** | **`8 ULP`** (bfloat16/float16), **`4 ULP`** (float32) |
| **Double Precision** | `float64` | **`1 ULP`** | **`4 ULP`** |

### Dynamic Caution Warning Banners

Whenever a user or agent selects a relaxed tolerance
($\text{RECOMMENDED} < \text{max\_allowed\_ulp} \le \text{HARD\_CEILING}$):

1. **CLI / Text Summary**: Prepends a prominent warning banner:

    ```
    ⚠️ CAUTION: A relaxed tolerance threshold (max_allowed_ulp=4) was configured.
    The recommended contract is <= 2 ULP for 'bfloat16' to guarantee numerical
    correctness. Ensure this elevation is analytically justified.
    ```
2. **Structured JSON Output**: Embeds a `"tolerance_audit"` block with
   `"is_relaxed_override": true` and the full `"caution_banner"`.
3. **Hard Ceiling Exception**: If `max_allowed_ulp > MAX_HARD_CEILING_ULP`,
   `validate_kernels` immediately raises a `ValueError`.

--------------------------------------------------------------------------------

## 5. Discrete, Indexing & Structured Mask Verification

For non-floating-point kernels (e.g., Mixture of Experts routing, embedding
lookups, ragged reductions, attention masking, and integer quantization
arithmetic), `numerical_generator` provides specialized discrete primitives:

### 1. Bounded Index Generation (MoE Expert IDs, Gather/Scatter)

Generates valid discrete indices strictly bounded in $[0,
\text{upper\_bound}-1]$ with deterministic injection of extreme boundary indices
($0$ and $\text{upper\_bound}-1$) to catch off-by-one errors:

```python
from xprof.cli.internal import numerical_generator

# Generate expert IDs in [0, 63] for MoE routing
expert_ids = numerical_generator.generate_index_tensor(
    shape=(batch_size, seq_len),
    upper_bound=64,
    lower_bound=0,
    dtype_str="int32",
    include_boundaries=True,
)
```

### 2. Monotonic Segment IDs (Ragged Reductions, Segmented Sum)

Generates non-decreasing segment IDs spanning $[0, \text{num\_segments}-1]$
along the reduction axis:

```python
# Generate sorted segment IDs for segmented_sum kernels
segment_ids = numerical_generator.generate_segment_ids_tensor(
    shape=(batch_size, total_tokens),
    num_segments=num_segments,
    is_sorted=True,
    dtype_str="int32",
)
```

### 3. Structured Mask Generation (Causal, Sparse, Padding)

Generates boolean masks for attention operators with multi-head $(B, H, S, S)$
broadcasting support:

```python
# 1. Causal lower-triangular mask (supports 4D broadcasting)
causal_mask = numerical_generator.generate_mask_tensor(
    shape=(batch_size, num_heads, seq_len, seq_len),
    mask_type="causal",
    dtype_str="bool",
)

# 2. Bernoulli sparse mask with target density p=0.3
sparse_mask = numerical_generator.generate_mask_tensor(
    shape=(batch_size, seq_len, seq_len),
    mask_type="bernoulli",
    density=0.3,
    dtype_str="bool",
)

# 3. Variable-length padding mask
padding_mask = numerical_generator.generate_mask_tensor(
    shape=(batch_size, max_seq_len),
    mask_type="padding",
    seq_lens=[128, 64, 256, 192],
    dtype_str="bool",
)
```

### 4. Integer Multi-Regime Suites & Exact Parity

When `dtype_str` is an integer type (`int32`, `int64`, `int16`, `int8`,
`uint32`, `uint8`) or `bool`:

*   `generate_test_suite` automatically generates small dynamic ranges, bounded
    indices, monotonic segment IDs, and extreme boundary limits (`0, 1, -1,
    \text{min\_val}, \text{max\_val}`).
*   `numerical_validator` enforces exact discrete delta $|y - \hat{y}| = 0$ with
    `max_allowed_ulp = 0`.

--------------------------------------------------------------------------------

## 6. ULP Interpretation & Root Cause Diagnostics

Empirical audits across accelerator workloads demonstrate that **histogram shape
does not uniquely classify defect types**. Benign reduction reassociation (e.g.
tree reduction vs sequential accumulation), accumulator downcasts (e.g. BF16
accumulation), and dropped/corrupted elements all produce broad ULP histograms
reaching thousands or millions of ULPs. The only defect pattern uniquely
distinguished by histogram shape is a pure 1-ULP spike caused by hardware
rounding mode differences (e.g. round-to-nearest-even vs round-towards-zero).

Use the following diagnostic framework instead of relying on ULP magnitude
alone:

| Observed Max ULP        | Diagnostic Assessment   | Recommended Action       |
| :---------------------- | :---------------------- | :----------------------- |
| **$0\text{--}1\text{    | **Exact Bitwise Parity  | Safe to merge for        |
: ULP}$**                 : / Rounding**\:          : refactors and pure       :
:                         : Bit-identical or at     : cleanups.                :
:                         : hardware rounding       :                          :
:                         : precision.              :                          :
| **$2\text{--}8\text{    | **Minor Numerical       | Verify $p_{99.9} \le     |
: ULPs}$**                : Drift**\: Typical of    : \text{contract}$; check  :
:                         : small                   : Q3 against float64       :
:                         : order-of-operation      : oracle to ensure         :
:                         : changes or fast-math    : candidate does not drift :
:                         : approximations.         : from ground truth.       :
| **$> 8\text{ ULPs}$ (to | **Ambiguous Broad       | **Do NOT automatically   |
: millions of ULPs)**     : Spectrum**\: Could be   : reject**\: Check         :
:                         : benign reduction        : `oracle_audit`. If       :
:                         : reassociation ($a +     : candidate is closer to   :
:                         : (b + c) \ne (a + b) +   : float64 oracle than      :
:                         : c$), accumulator        : baseline, candidate is   :
:                         : precision loss, or      : more accurate. If        :
:                         : algorithmic regression. : baseline and candidate   :
:                         :                         : both drift from oracle,  :
:                         :                         : inspect accumulator      :
:                         :                         : dtype.                   :
| **Inf / NaN**           | **Algebraic Instability | Inspect scale factors,   |
:                         : / Exponent Overflow**\: : normalizations (e.g.     :
:                         : Division by zero,       : $x - \max(x)$), and      :
:                         : un-normalized Softmax,  : subnormal clamping.      :
:                         : or missing scale        :                          :
:                         : factor.                 :                          :

### Decision Rubric for Elevated ULPs

1.  **Refactor / Cleanup Tasks**: If your task was a refactor (renaming, layout
    change, plumbing), **any** elevated ULP ($> \text{contract}$) is an
    unexpected regression. Investigate immediately.
2.  **Kernel Optimization Tasks (Reassociation / Tiling / Fusion)**: Large ULP
    deviations are expected when reduction order changes.
    *   Compare `candidate_max_ulp_from_oracle` with
        `reference_max_ulp_from_oracle`.
    *   If candidate distance $\le$ reference distance, the candidate is
        mathematically equal or superior to the reference despite failing Q1 ULP
        parity.
    *   Run downstream end-to-end unit and integration tests (or convergence
        benchmarks) rather than gating on isolated ULP parity.

--------------------------------------------------------------------------------

## 7. Test Suite Persistence & Remote Hardware Execution

To guarantee identical test distributions between local agent workspaces and
physical accelerator testbeds (e.g. TPU v5e/v5p/v6e, GPUs, custom ASICs),
test suites are persisted to `.npz` archive files with embedded JSON metadata
manifests:

```python
from xprof.cli.internal import numerical_generator

# Generate and persist test suite fixture
suite = numerical_generator.generate_test_suite(
    shapes=[(16, 128, 128)],
    dtype_str="bfloat16",
    tier="presubmit",
    seed=42,
)
numerical_generator.save_test_suite(suite, "/tmp/rms_norm_golden.npz")

# Load fixture on accelerator / physical testbed
loaded_suite = numerical_generator.load_test_suite("/tmp/rms_norm_golden.npz")
```

### JSON Manifest Properties

*   `schema_version`: Data structure compatibility (`1.0.0`).
*   `sha256_hash`: Bitwise integrity hash of all array tensors.
*   `generation_timestamp`: ISO 8601 creation time.
*   `tier`, `dtype_str`, `shapes`: Full configuration reproduction parameters.

--------------------------------------------------------------------------------

## 8. Mathematical Foundations & Heavy-Tail Theory (Deep Dive)

### The Tolerance Dilemma

Standard unit testing approaches for ML kernels frequently fail in two opposite
ways:

```
+-----------------------------------------------------------------------------+
|                           THE TOLERANCE DILEMMA                             |
|                                                                             |
|  [Loose Tolerances: rtol=1e-1, atol=1e-1]                                   |
|   -> Hides catastrophic accumulator truncation, broken reductions, and      |
|      subnormal cancellation.                                                |
|                                                                             |
|  [Strict Tolerances: rtol=1e-6, atol=1e-6]                                  |
|   -> Fails 100% of valid BF16/FP8 kernels due to hardware machine epsilon   |
|      (BF16 eps ~ 7.8e-3; FP8 eps ~ 0.125).                                 |
|                                                                             |
|  [Standard Gaussian Inputs: N(0, 1)]                                        |
|   -> 99.7% of values fall within [-3, 3]. Completely misses Softmax exp(x)  |
|      overflow (x >= 88.72 in FP32), FP8 saturation (>448), and subnormals.  |
+-----------------------------------------------------------------------------+
```

### Statistical Foundations & Input Regimes

Real-world transformer activations in attention projection and MLP intermediate
layers exhibit heavy-tailed power-law decay ($P(|X| > x) \sim x^{-\nu}$ with
$\nu \approx 2.5\text{--}4.0$). Standard Gaussian noise ($\mathcal{N}(0, 1)$)
does not excite the non-linear tail saturation regions of hardware
floating-point units.

#### The Five Input Regimes

1.  **Standard Normal Regime (`normal_batch_0`)**: Standard baseline
    distribution executed by default for rapid single-batch verification
    ($m=1$).
2.  **Student's t Distribution ($\nu \in [2.5, 4.0]$)**: Power-law decay with
    heavy tails, generating activation spreads that stress exponent dynamics.
3.  **Localized Activation Spikes ($10\times\text{--}100\times$)**: Injected
    outliers matching empirically observed LLM outlier feature dimensions.
4.  **Catastrophic Cancellation Pairs**: Alternating $\pm M$ pairs with small
    $\epsilon$ residuals, detecting coarse accumulator downcasting.
5.  **Hardware Memory / Register Tile Boundary Strides**: Subnormals, minimum
    normals, and zeros aligned to hardware tile / vector register / SIMD / VMEM
    / warp boundaries (e.g., 128-element strides).

> [!NOTE] **Dynamic Range Reach**: Empirical measurements show standard
> generated `float32` suites reach peak values around $|x| \approx 59.36$. While
> this stresses exponent variation and dynamic range substantially better than
> standard Gaussians ($[-3, 3]$), it does not naturally reach float32 $\exp(x)$
> overflow ($x \ge 88.72$) or FP8 e4m3 saturation ($448.0$) without explicit
> operator-specific target scaling.

#### Finite vs. Infinite Moment Regimes

*   **Finite 4th Moment ($\nu > 4.0$, e.g., $\nu = 5, 6, 8, 10$)**: Theoretical
    variance $\sigma^2 = \frac{\nu}{\nu - 2}$ and 4th moment $\mu_4 =
    \frac{3\nu^2}{(\nu-2)(\nu-4)}$ are finite.
*   **Infinite 4th Moment ($\nu \le 4.0$, e.g., $\nu = 2.5, 3.0, 4.0$)**:
    $\mu_4 = \infty \implies \text{Var}(S^2) = \infty$. Testing sample variance
    $S^2$ with a fixed `rtol` is mathematically ill-posed due to extreme tail
    draws.
*   **Infinite 3rd Moment ($\nu \le 3.0$)**: $\mu_3 = \infty \implies$ Sample
    moment skewness has infinite variance.

#### Robust Non-Parametric Metrics for Heavy Tails

To validate heavy-tailed tensors reliably without variance flakiness, `xprof`
evaluates:

1.  **Theoretical Quantiles**: Non-parametric percentiles ($Q_{25}, Q_{50},
    Q_{75}, Q_{90}, Q_{95}$) derived from Student's t quantile function
    $F^{-1}(p; \nu)$.
2.  **Interquartile Range**: $\text{IQR} = Q_{75} - Q_{25} = 2 \cdot Q_{75}$
    (for symmetric zero-mean distributions).
3.  **Bowley's Quantile Skewness**:
    $$S_{\text{Bowley}} = \frac{Q_{75} + Q_{25} - 2Q_{50}}{Q_{75} - Q_{25}}$$
    Bounded in $[-1, 1]$ and equals $0.0$ for any symmetric distribution.

### Floating-Point Bit Layouts & Machine Epsilon

Data Type      | Sign | Exp | Mant | Machine Epsilon       | Min Subnormal          | Max Finite            | Safe ULP Bound
:------------- | :--: | :-: | :--: | :-------------------: | :--------------------: | :-------------------: | :------------:
**`float64`**  | 1    | 11  | 52   | $2.22 \times 10^{-16}$| $4.94 \times 10^{-324}$| $1.80 \times 10^{308}$| $\le 1\text{ ULP}$
**`float32`**  | 1    | 8   | 23   | $1.19 \times 10^{-7}$ | $1.40 \times 10^{-45}$ | $3.40 \times 10^{38}$ | $\le 1\text{--}2\text{ ULPs}$
**`bfloat16`** | 1    | 8   | 7    | $7.81 \times 10^{-3}$ | $9.18 \times 10^{-41}$ | $3.39 \times 10^{38}$ | $\le 1\text{--}2\text{ ULPs}$
**`float16`**  | 1    | 5   | 10   | $9.77 \times 10^{-4}$ | $5.96 \times 10^{-8}$  | $65,504.0$            | $\le 1\text{--}2\text{ ULPs}$
**`fp8_e4m3`** | 1    | 4   | 3    | $0.125$               | $1.95 \times 10^{-3}$  | $448.0$               | $\le 1\text{--}2\text{ ULPs}$
**`fp8_e5m2`** | 1    | 5   | 2    | $0.250$               | $1.53 \times 10^{-5}$  | $57,344.0$            | $\le 1\text{--}2\text{ ULPs}$

### Continuous Integer Mapping & IEEE-754 Edge Cases

An **ULP (Unit in the Last Place)** measures the discrete number of
representable floating-point steps between two values. `numerical_validator`
maps sign-magnitude bit patterns to a continuous signed integer index:

$$I(x) = \begin{cases}
+\text{magnitude\_bits}(x) & \text{if } x \ge 0 \\
-\text{magnitude\_bits}(x) & \text{if } x < 0
\end{cases}$$

The ULP distance between actual output $y$ and expected reference $\hat{y}$ is:

$$\text{ULP}(y, \hat{y}) = |I(y) - I(\hat{y})|$$

1.  **Signed Zeros ($+0.0$ vs. $-0.0 \to 0\text{ ULP}$)**: $+0.0$ (`0x0000`) and
    $-0.0$ (`0x8000`) both map to integer index $0$, preventing artificial error
    jumps across zero.
2.  **Cross-Zero Subnormal Transitions**: Smallest positive subnormal ($+1$) vs.
    smallest negative subnormal ($-1$) evaluates to $|(+1) - (-1)| =
    \mathbf{2\text{ ULP}}$.
3.  **Normal-to-Subnormal Transitions**: Minimum normal (bits `0x0080` in bf16,
    index $128$) vs. maximum subnormal (bits `0x007F`, index $127$) evaluates to
    exact $\mathbf{1\text{ ULP}}$.
4.  **Scale Invariance**: The spacing of $1\text{ ULP}$ is identical at
    $10^{-30}$, $1.0$, and $10^{+30}$, unlike scalar absolute tolerance
    $\text{atol}$ which degrades across magnitudes.
