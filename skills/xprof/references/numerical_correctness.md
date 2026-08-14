# Numerical Correctness & Heavy-Tailed Verification in XProf

This guide provides the complete methodology for verifying numerical equivalence
between baseline and optimized TPU/Pallas kernels and compiler transformations.

--------------------------------------------------------------------------------

## 1. The Tolerance Dilemma

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

### The Solution: Multi-Regime Testing & Bit-Level ULP Distance

Instead of ungrounded scalar tolerances, `xprof` uses:

1.  **Multi-Regime Heavy-Tailed Input Generation**: Student's t distributions
    ($\nu \in [2.5, 4.0]$), localized activation spikes
    ($10\times\text{--}100\times$), cancellation pairs, and tile-aware boundary
    strides.
2.  **Unit in the Last Place (ULP) Bitwise Metrics**: Discrete integer steps
    between floating-point bit patterns, providing exact hardware-aligned error
    bounds regardless of dtype.

--------------------------------------------------------------------------------

## 2. Statistical Foundations & Heavy-Tail Theory

Real-world transformer activations in attention projection and MLP intermediate
layers exhibit heavy-tailed power-law decay ($P(|X| > x) \sim x^{-\nu}$ with
$\nu \approx 2.5\text{--}4.0$). Standard Gaussian noise ($\mathcal{N}(0, 1)$)
never excites the non-linear tail saturation regions of hardware floating-point
units.

### The Four Input Regimes

1.  **Student's t Distribution ($\nu \in [2.5, 4.0]$)**: Power-law decay with
    heavy tails, generating activation spreads that stress exponent dynamics.
2.  **Localized Activation Spikes ($10\times\text{--}100\times$)**: Injected
    outliers matching empirically observed LLM outlier feature dimensions.
3.  **Catastrophic Cancellation Pairs**: Alternating $\pm M$ pairs with small
    $\epsilon$ residuals, detecting coarse accumulator downcasting.
4.  **TPU VMEM Tile Boundary Strides**: Subnormals, minimum normals, and zeros
    aligned to 128-element TPU VMEM subcore boundaries.

### Finite vs. Infinite Moment Regimes

The statistical behavior of Student's t distribution depends critically on the
degrees of freedom $\nu$:

*   **Finite 4th Moment ($\nu > 4.0$, e.g., $\nu = 5, 6, 8, 10$)**: Theoretical
    variance $\sigma^2 = \frac{\nu}{\nu - 2}$ and 4th moment $\mu_4 =
    \frac{3\nu^2}{(\nu-2)(\nu-4)}$ are finite. The sample variance $S^2$ has
    Relative Standard Error $\text{RSE}(S^2) \approx \sqrt{\frac{1}{N}
    \left(\frac{6}{\nu-4} + 2\right)}$.
*   **Infinite 4th Moment ($\nu \le 4.0$, e.g., $\nu = 2.5, 3.0, 4.0$)**:
    $\mu_4 = \infty \implies \text{Var}(S^2) = \infty$. Testing sample variance
    $S^2$ with a fixed `rtol` is mathematically ill-posed due to extreme tail
    draws.
*   **Infinite 3rd Moment ($\nu \le 3.0$)**: $\mu_3 = \infty \implies$ Sample
    moment skewness has infinite variance.

### Robust Non-Parametric Metrics for Heavy Tails

To validate heavy-tailed tensors reliably without variance flakiness, `xprof`
evaluates:

1.  **Theoretical Quantiles**: Non-parametric percentiles ($Q_{25}, Q_{50},
    Q_{75}, Q_{90}, Q_{95}$) derived from Student's t quantile function
    $F^{-1}(p; \nu)$.
2.  **Interquartile Range**: $\text{IQR} = Q_{75} - Q_{25} = 2 \cdot Q_{75}$
    (for symmetric zero-mean distributions).
3.  **Bowley's Quantile Skewness**: $$S_{\text{Bowley}} = \frac{Q_{75} +
    Q_{25} - 2Q_{50}}{Q_{75} - Q_{25}}$$ Bounded in $[-1, 1]$ and equals $0.0$
    for any symmetric distribution.

--------------------------------------------------------------------------------

## 3. Floating-Point Bit Layouts & Machine Epsilon

Data Type      | Sign | Exp | Mant | Machine Epsilon       | Min Subnormal          | Max Finite            | Safe ULP Bound
:------------- | :--: | :-: | :--: | :-------------------: | :--------------------: | :-------------------: | :------------:
**`float32`**  | 1    | 8   | 23   | $1.19 \times 10^{-7}$ | $1.40 \times 10^{-45}$ | $3.40 \times 10^{38}$ | $\le 1\text{--}2\text{ ULPs}$
**`bfloat16`** | 1    | 8   | 7    | $7.81 \times 10^{-3}$ | $9.18 \times 10^{-41}$ | $3.39 \times 10^{38}$ | $\le 1\text{--}2\text{ ULPs}$
**`float16`**  | 1    | 5   | 10   | $9.77 \times 10^{-4}$ | $5.96 \times 10^{-8}$  | $65,504.0$            | $\le 1\text{--}2\text{ ULPs}$
**`fp8_e4m3`** | 1    | 4   | 3    | $0.125$               | $1.95 \times 10^{-3}$  | $448.0$               | $\le 1\text{--}2\text{ ULPs}$
**`fp8_e5m2`** | 1    | 5   | 2    | $0.250$               | $1.53 \times 10^{-5}$  | $57,344.0$            | $\le 1\text{--}2\text{ ULPs}$

--------------------------------------------------------------------------------

## 4. ULP Distance Mechanics & IEEE-754 Edge Cases

An **ULP (Unit in the Last Place)** measures the discrete number of
representable floating-point steps between two values.

### Continuous Integer Mapping

Floating-point numbers use IEEE-754 sign-magnitude encoding.
`numerical_validator` maps sign-magnitude bit patterns to a continuous signed
integer index:

$$I(x) = \begin{cases}
+\text{magnitude\_bits}(x) & \text{if } x \ge 0 \\
-\text{magnitude\_bits}(x) & \text{if } x < 0
\end{cases}$$

The ULP distance between actual output $y$ and expected reference $\hat{y}$ is:

$$\text{ULP}(y, \hat{y}) = |I(y) - I(\hat{y})|$$

### Canonical Edge Cases

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

--------------------------------------------------------------------------------

## 5. 4-Way Kernel Parity Topology

In TPU kernel development and compiler optimization (e.g., Rosetta vs. Pallas),
numerical equivalence is established across a **4-Way Verification Matrix**:

```
                  +-----------------------------------+
                  |   A: JAX Golden Reference Math   |
                  +-----------------------------------+
                                    |
                                    | (<= 2 ULP baseline)
                                    v
                  +-----------------------------------+
                  |    B: Pallas Baseline Kernel      |
                  +-----------------------------------+
                                    |
                                    | (<= 1 ULP optimization)
                                    v
                  +-----------------------------------+
                  |    C: Pallas Optimized Kernel     |
                  +-----------------------------------+
                                    |
                                    | (0 ULP bitwise translation)
                                    v
                  +-----------------------------------+
                  |   D: Rosetta Autotuned Kernel     |
                  +-----------------------------------+
```

### Acceptance Matrix

| Comparison      | Target Parity | Acceptance Criteria | Rationale          |
| :-------------- | :-----------: | :-----------------: | :----------------- |
| **$A            | Baseline Math | $\le 2\text{ ULP}$  | Validates custom   |
: \leftrightarrow :               : ($p_{99.9} \le 1$)  : kernel against     :
: B$**            :               :                     : high-precision JAX :
:                 :               :                     : reference.         :
| **$B            | Optimization  | $\le 1\text{ ULP}$  | Confirms memory    |
: \leftrightarrow :               : ($p_{99.9} \le 1$)  : tiling and loop    :
: C$**            :               :                     : unrolling preserve :
:                 :               :                     : precision.         :
| **$C            | Compiler      | $\mathbf{0\text{    | Confirms           |
: \leftrightarrow : Translation   : ULP}}$ (Bitwise     : compiler-generated :
: D$**            :               : Match)              : custom call        :
:                 :               :                     : matches hand-tuned :
:                 :               :                     : Pallas.            :

--------------------------------------------------------------------------------

## 6. Test Suite Persistence & Remote Hardware Execution

To guarantee identical test distributions between local agent workspaces and
physical TPU accelerators (e.g. TPU v5e/v5p/v6e), test suites are persisted to
`.npz` archive files with embedded JSON metadata manifests:

```python
from google3.third_party.xprof.plugin.xprof.cli.internal import numerical_generator

# Generate and persist test suite fixture
suite = numerical_generator.generate_test_suite(
    shape=(16, 128, 128),
    dtype_str="bfloat16",
    tier="presubmit",
    seed=42,
)
numerical_generator.save_test_suite(suite, "/tmp/rms_norm_golden.npz")

# Load fixture on physical TPU testbed
loaded_suite = numerical_generator.load_test_suite("/tmp/rms_norm_golden.npz")
```

### JSON Manifest Properties

*   `schema_version`: Data structure compatibility (`1.0.0`).
*   `sha256_hash`: Bitwise integrity hash of all array tensors.
*   `generation_timestamp`: ISO 8601 creation time.
*   `tier`, `dtype_str`, `shapes`: Full configuration reproduction parameters.

--------------------------------------------------------------------------------

## 7. Sample Size Determination: Operational Testing Tiers

Tier               | Total Tensors ($m$)      | Composition                                                        | Latency                       | Recommended Use
:----------------- | :----------------------: | :----------------------------------------------------------------- | :---------------------------: | :--------------
**`fast_agent`**   | **$m = 5$**              | 2 Student-t + 1 Outlier ($50\times$) + 1 Cancellation + 1 Boundary | $\sim 1\text{--}2\text{ s}$   | Interactive pair-programming iteration by agent
**`presubmit`**    | **$m = 12\text{--}15$**  | 6-8 Student-t + 3-4 Outliers + 3 Boundary probes                   | $\sim 5\text{--}8\text{ s}$   | Automated presubmit before submitting CL
**`deep_fuzzing`** | **$m = 50\text{--}100$** | 30 Student-t + 15 Outliers + 5 Boundary grids                      | $\sim 30\text{--}60\text{ s}$ | Compiler pass / Pallas release qualification

--------------------------------------------------------------------------------

## 8. CLI Tool Usage (`xprof verify_numerical_parity`)

```bash
# Verify parity using the fast_agent tier
xprof verify_numerical_parity \
  --kernel_ref="my_module.reference_fn" \
  --kernel_candidate="my_module.optimized_fn" \
  --shapes="[16, 1024]" \
  --dtype_str="bfloat16" \
  --tier="fast_agent"
```

### JSON Output Format

```json
{
  "is_numerically_equivalent": true,
  "overall_max_ulp": 1,
  "failed_batches_count": 0,
  "total_batches_count": 5,
  "summary_message": "PASSED: Kernels are numerically equivalent across 5 batches (Max ULP: 1, Limit: 2).",
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

--------------------------------------------------------------------------------

## 9. Python API Usage for Kernel Optimization Agents

```python
from google3.third_party.xprof.plugin.xprof.cli.internal import numerical_generator
from google3.third_party.xprof.plugin.xprof.cli.internal import numerical_validator

# 1. Define Reference and Candidate Kernels
def ref_kernel(a, b):
  return jnp.dot(a, b)

def pallas_kernel(a, b):
  # Optimized custom TPU Pallas implementation
  return custom_pallas_matmul(a, b)

# 2. Validate Parity Across Full Multi-Regime Suite
report = numerical_validator.validate_kernels(
    kernel_ref=ref_kernel,
    kernel_candidate=pallas_kernel,
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
      print(f"  Batch {batch.batch_name} ({batch.regime}): Max ULP={batch.max_ulp_distance}")
```

--------------------------------------------------------------------------------

## 10. ULP Interpretation & Root Cause Diagnostics

| Observed Max ULP       | Diagnostic Assessment    | Recommended Remedy      |
| :--------------------- | :----------------------- | :---------------------- |
| **$0\text{--}1\text{   | **Exact Numerical        | Optimization is safe to |
: ULP}$**                : Parity**. Hardware       : merge.                  :
:                        : rounding bit-accuracy    :                         :
:                        : confirmed.               :                         :
| **$2\text{--}4\text{   | **Legitimate             | Check $p_{99.9} \le 2$; |
: ULPs}$**               : Associativity            : acceptable if within    :
:                        : Reordering**. Tree       : analytical bounds.      :
:                        : reduction or FMA fusion. :                         :
| **$10\text{--}50\text{ | **Accumulator Precision  | Keep reduction          |
: ULPs}$**               : Truncation**.            : accumulator in FP32     :
:                        : Intermediate downcast    : before final cast.      :
:                        : (e.g. BF16 accumulation  :                         :
:                        : instead of FP32).        :                         :
| **$> 1000\text{ ULPs}$ | **Algebraic Instability  | Inspect Softmax         |
: / Inf**                : / Exponent Saturation**. : normalization and scale :
:                        : Missing $x - \max(x)$ or : factors.                :
:                        : improper scale factor.   :                         :
