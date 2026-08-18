# Numerical Correctness Verification in XProf

This guide provides the complete operational methodology and reference for
verifying numerical equivalence between baseline reference implementations and
optimized compute kernels (TPU, GPU, CPU, Pallas, Triton, C++, CUDA) across
compiler transformations, lowerings, and autotuning.

--------------------------------------------------------------------------------

## 1. Overview & Situational Use

Numerical verification evaluates mathematical parity between a **Reference
Callable ($f_{\text{ref}}$)** (e.g., high-precision math, trusted baseline, or
unoptimized implementation) and an **Optimized Candidate Callable
($f_{\text{cand}}$)** (e.g., tiled Pallas kernel, Triton op, or compiler custom
call):

$$\Delta_{\text{ULP}}(x) =
  \text{ULP\_Distance}\left(f_{\text{cand}}(x), f_{\text{ref}}(x)\right)$$

Verification succeeds if
$\max_{x \in \mathcal{T}} \Delta_{\text{ULP}}(x) \le \text{max\_allowed\_ulp}$
across all test batches $\mathcal{T}$.

### Situational Verification Matrix

Workload / Domain              | Target Operator & Use Case                     | Verification Approach
:----------------------------- | :--------------------------------------------- | :--------------------
**Continuous Floating-Point**  | MatMul, FlashAttention, RMSNorm, Activations   | Multi-regime heavy-tailed ULP verification (Student-t, activation spikes, cancellation pairs, VMEM strides).
**Discrete / Token Routing**   | MoE Expert Routing, Embedding Lookups, Gather  | Bounded index generation with deterministic boundary extreme pinning ($0, N-1$).
**Ragged Reductions**          | `segmented_sum`, ragged token batch reductions | Monotonic non-decreasing segment ID partitioning along reduction axis.
**Attention Topologies**       | Causal, Sparse, and Sequence Padding Masks     | Structured boolean mask generation with 4D $(B, H, S, S)$ multi-head attention broadcasting.
**Quantized Arithmetic**       | INT8/INT4 quantization, integer accumulators   | Integer multi-regime suites testing extreme boundaries (`min_val`, `max_val`, $0, 1, -1$).

--------------------------------------------------------------------------------

## 2. Quick Start Workflows

### Workflow A: CLI Tool (`xprof verify_numerical_parity`)

⚠️ **MANDATORY**: When asked to verify numerical parity or compare reference
and candidate functions, you **MUST** run the `xprof verify_numerical_parity`
CLI command (or `xprof_cli verify_numerical_parity`) rather than writing a
custom inline script.

```bash
# Verify parity between two Python callables using the fast_agent tier
xprof verify_numerical_parity \
  --kernel_ref="my_module.reference_fn" \
  --kernel_candidate="my_module.optimized_fn" \
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
  "total_batches_count": 5,
  "summary_message": "PASSED: Kernels are numerically equivalent across 5 batches (Max ULP: 1, Limit: 2).",
  "tolerance_audit": {
    "recommended_contract_ulp": 2,
    "configured_max_ulp": 2,
    "hard_safety_ceiling": 8,
    "is_relaxed_override": false,
    "caution_banner": null
  },
  "batch_results": [
    {
      "batch_name": "student_t_batch_0",
      "regime": "student_t",
      "max_ulp_distance": 1,
      "p99_9_ulp_distance": 1.0,
      "mean_ulp_distance": 0.04,
      "ulp_histogram": {
        "<=1_ulp": 16384,
        "<=2_ulp": 16384,
        ">2_ulp": 0
      },
      "has_nan_or_inf": false,
      "passed": true
    }
  ]
}
```

### Workflow B: Python API (`validate_kernels`)

For programmatic integration within Python test harnesses or optimization loops:

```python
from google3.third_party.xprof.plugin.xprof.cli.internal import numerical_validator

# 1. Define Reference and Candidate Kernels
def ref_kernel(a, b):
  return jnp.dot(a, b)

def candidate_kernel(a, b):
  return custom_optimized_matmul(a, b)

# 2. Validate Parity Across Full Multi-Regime Suite
report = numerical_validator.validate_kernels(
    kernel_ref=ref_kernel,
    kernel_candidate=candidate_kernel,
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

## 3. Bitwise ULP Tolerance Contracts & Anti-Abuse Guardrails

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

## 4. Discrete, Indexing & Structured Mask Verification

For non-floating-point kernels (e.g., Mixture of Experts routing, embedding
lookups, ragged reductions, attention masking, and integer quantization
arithmetic), `numerical_generator` provides specialized discrete primitives:

### 1. Bounded Index Generation (MoE Expert IDs, Gather/Scatter)

Generates valid discrete indices strictly bounded in $[0,
\text{upper\_bound}-1]$ with deterministic injection of extreme boundary indices
($0$ and $\text{upper\_bound}-1$) to catch off-by-one errors:

```python
from google3.third_party.xprof.plugin.xprof.cli.internal import numerical_generator

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

## 5. ULP Interpretation & Root Cause Diagnostics

| Observed Max ULP       | Diagnostic Assessment    | Recommended Remedy      |
| :--------------------- | :----------------------- | :---------------------- |
| **$0\text{--}1\text{   | **Exact Numerical        | Optimization is safe to |
: ULP}$**                : Parity**. Hardware       : merge.                  :
:                        : rounding bit-accuracy    :                         :
:                        : confirmed.               :                         :
| **$2\text{--}4\text{   | **Legitimate             | Check $p_{99.9} \le 2$; |
: ULPs}$**               : Associativity            : acceptable if within    :
:                        : Reordering**. Tree       : analytical bounds       :
:                        : reduction or FMA fusion. : (e.g. Split-K).         :
| **$10\text{--}50\text{ | **Accumulator Precision  | Keep reduction          |
: ULPs}$**               : Truncation**.            : accumulator in FP32     :
:                        : Intermediate downcast    : before final cast.      :
:                        : (e.g. BF16 accumulation  :                         :
:                        : instead of FP32).        :                         :
| **$> 1000\text{ ULPs}$ | **Algebraic Instability  | Inspect Softmax         |
: / Inf**                : / Exponent Saturation**. : normalization and scale :
:                        : Missing $x - \max(x)$ or : factors.                :
:                        : improper scale factor.   :                         :

--------------------------------------------------------------------------------

## 6. Test Suite Persistence & Remote Hardware Execution

To guarantee identical test distributions between local agent workspaces and
physical accelerator testbeds (e.g. TPU v5e/v5p/v6e, GPUs, custom ASICs),
test suites are persisted to `.npz` archive files with embedded JSON metadata
manifests:

```python
from google3.third_party.xprof.plugin.xprof.cli.internal import numerical_generator

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

## 7. Mathematical Foundations & Heavy-Tail Theory (Deep Dive)

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

### Statistical Foundations & Heavy Tails

Real-world transformer activations in attention projection and MLP intermediate
layers exhibit heavy-tailed power-law decay ($P(|X| > x) \sim x^{-\nu}$ with
$\nu \approx 2.5\text{--}4.0$). Standard Gaussian noise ($\mathcal{N}(0, 1)$)
never excites the non-linear tail saturation regions of hardware floating-point
units.

#### The Four Input Regimes

1.  **Student's t Distribution ($\nu \in [2.5, 4.0]$)**: Power-law decay with
    heavy tails, generating activation spreads that stress exponent dynamics.
2.  **Localized Activation Spikes ($10\times\text{--}100\times$)**: Injected
    outliers matching empirically observed LLM outlier feature dimensions.
3.  **Catastrophic Cancellation Pairs**: Alternating $\pm M$ pairs with small
    $\epsilon$ residuals, detecting coarse accumulator downcasting.
4.  **Hardware Memory / Register Tile Boundary Strides**: Subnormals, minimum
    normals, and zeros aligned to hardware tile / vector register / SIMD / VMEM
    / warp boundaries (e.g., 128-element strides).

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
