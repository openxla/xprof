# SQNR Contract (`contract="sqnr"`)

SQNR (signal-to-quantization-noise ratio) measures the total error energy of a
candidate against a reference, in dB. All dB values use

$$S(a, b) = 10 \log_{10} \frac{\sum b^2}{\sum (a - b)^2}$$

with $a$ the output under test and $b$ the baseline. Higher is better. Every
6.02 dB is about one significand bit.

## Choose the Contract

Contract        | Use it when                                                                                                       | Gate
--------------- | ----------------------------------------------------------------------------------------------------------------- | ----
`ulp` (default) | The candidate must match the reference up to rounding: fusion, layout, same-precision rewrites.                   | Max ULP per element.
`bitwise`       | The bits must be identical (sign of zero, NaN payloads).                                                          | Bit-pattern equality.
`row_rel_l2`    | The kernel spec states a relative-L2 tolerance.                                                                   | Whole-output and per-row relative L2.
`sqnr`          | The kernel is deliberately lossy (FP8, FP4, int8, bf16 inputs) and the budget is in dB, or no tolerance is given. | SQNR floor, oracle rule, anti-cheat checks.

SQNR and relative L2 are the same quantity on different scales: $\text{SQNR} =
-20 \log_{10}(\text{rel\_l2})$. A `rel_l2_tol` of 2e-2 is 34.0 dB. 30 dB is a
`rel_l2` of 0.0316. 40 dB is 0.01.

Every contract reports the SQNR fields. Only `contract="sqnr"` gates on them.
Under `ulp`, `bitwise` and `row_rel_l2`, the SQNR report is report only and
non-strict: it has the global and oracle fields, and runs no strict checks.

## Run It

The `verify` command and the `validate_kernels` call are in the SKILL.md Quick
Start. This section lists the SQNR flags and the lower-level functions.

Flag            | Default   | Meaning
--------------- | --------- | -------
`--min_sqnr_db` | `auto`    | Global floor in dB, or `auto` (see below). Here `auto` models only the output dtype: a kernel with narrow inputs or a narrow accumulator needs an explicit value. Under `six_gate_v1`, an explicit `auto` keeps this dtype-derived floor, while an omitted flag gives 45 dB.
`--sqnr_alpha`  | `1e-6`    | False-alarm rate per output on honest noise that the noise model fits. The strict check bounds come from it. Two float32 summation orders of one bf16 matmul gave 0 false failures in 620 seeds (K up to 4096, outputs up to 1048576 elements; 95% upper bound about 1 in 200 leaves).
`--sqnr_preset` | `derived` | `derived`, or `six_gate_v1` (fixed constants, 45 dB default floor).
`--sqnr_strict` | `True`    | `False` runs only the global floor and the oracle rule. The default is `True` under `--contract=sqnr`, else `False`.

`replay` takes the same flags as overrides of the dump's stored config. An
omitted flag keeps the stored value. With `--sqnr_preset`, a stored floor equal
to the stored preset's default takes the new preset's default.

```python
from xprof.xparity import sqnr

# Report-only metrics for two arrays (no anti-cheat checks):
leaf = sqnr.compute_sqnr(
    candidate_out,
    reference_out,
    oracle=float64_out,  # Optional.
    precision=sqnr.PrecisionSpec(
        operand_dtypes=("float8_e4m3fn", "float8_e4m3fn"),
        output_dtype="bfloat16",
    ),
)
print(leaf.sqnr_db, leaf.threshold_db, leaf.threshold_source)

# Strict checks with a full precision model for two arrays:
leaf = sqnr.audit_leaf(
    [(reference_out, candidate_out)],
    precision=sqnr.precision_from_dtypes(
        reference_out.dtype,
        candidate_out.dtype,
        input_dtypes=("float8_e4m3fn", "float8_e4m3fn"),
    ),
    input_views=[[x]],  # Optional: inputs with the output's shape.
    oracle_pairs=[float64_out],  # Optional.
)
print(leaf.passed, leaf.threshold_db, leaf.failed_reasons)
```

`sqnr.SqnrConfig.from_options(min_sqnr_db=..., alpha=..., strict=...,
preset=...)` builds the same config as the CLI flags. An omitted `min_sqnr_db`
takes the preset default (`auto` under `derived`, 45 dB under `six_gate_v1`).
An explicit `"auto"` keeps the dtype-derived floor under every preset.
`sqnr_config=None` in `validate_kernels` means `sqnr.SqnrConfig()` under
`contract="sqnr"`, and
`sqnr.SqnrConfig(strict=False)` under the other contracts.
`sqnr.SqnrConfig.six_gate_v1()` selects the fixed-constant preset.

## How `auto` Sets the Floor

`sqnr.expected_sqnr_db(precision)` adds the noise power of each rounding source
$i$ of a `sqnr.PrecisionSpec`, with per-rounding SQNR $S_i$:

$$E = -10 \log_{10} \sum_i 10^{-S_i/10}$$

-   The sources are the operand dtypes, the output dtype and the accumulator
    dtype of the spec. A source counts only if it is narrower than float32.
-   `xparity verify`, `replay` and `validate_kernels` build the spec from
    the output dtypes only. The one source is the candidate output dtype: a
    float32 output gives the 45 dB default, and a bf16 output gives 49.6 dB.
    They do not see the input dtypes, the accumulator or $K$, and they add no
    integer warning. For a kernel with narrow inputs (int8, int4, FP8, FP4,
    bf16) or a narrow accumulator, pass an explicit `--min_sqnr_db` from the
    kernel spec, or compute the modeled floor with `sqnr.resolve_threshold`
    (below).
-   `sqnr.precision_from_dtypes(reference_dtype, candidate_dtype,
    input_dtypes=..., accumulator_dtype=..., reduction_length=...)` builds a
    full spec. It keeps the inputs narrower than float32 and takes the output
    source from the candidate dtype.
-   `PrecisionSpec.output_dtype` defaults to None, which adds no output source.
    `compute_sqnr` and `audit_leaf` use the reference dtype only when their
    `precision` argument is None. bf16 x bf16 operands give 52.6 dB with
    `output_dtype=None`, and 50.8 dB with `output_dtype="bfloat16"`.
-   For a plain float format, $S = 6.02\,p + 7.44$ dB. $p$ is the significand
    width including the hidden bit (bf16 8, fp16 11, e4m3 4, e5m2 3).
-   For block-scaled formats and `float4_e2m1fn`, $S$ is a simulated value
    (table below). Pass the format name (for example `"mxfp4"`) in
    `input_dtypes` or `PrecisionSpec.operand_dtypes`.
-   An accumulator narrower than float32 with reduction length $K > K_0$ adds
    the source $S_{acc} - 10 \log_{10}(K/K_0)$, with $K_0 = 2$. A bf16
    accumulator with $K = 4096$ is a 22.5 dB source by itself.
-   The floor is $E - 6$ dB (`auto_margin_db`).
-   If $E$ minus `auto_margin_db` (6 dB by default) is not positive, `auto`
    raises a ValueError. Pass an explicit `min_sqnr_db`. With a bf16
    accumulator and no other source, this happens from $K = 182{,}403$.
-   `PrecisionSpec.reduction_length` must be None or an int >= 1.
-   With no source narrower than float32, the floor is 45 dB.
-   `auto` does not model narrow integer formats (int16, int8, int4). With such
    a format as an operand, output or accumulator dtype in the spec, the floor
    falls back to 45 dB and the report adds a `warnings` entry. Always pass an
    explicit floor from the kernel spec for integer quantization. int32 and
    int64 inputs (indices, lengths) and bool masks do not change the floor. The
    warning is left out on a discrete leaf (integer or bool reference or
    candidate), because that leaf needs an exact match.

```python
floor_db, source = sqnr.resolve_threshold(
    sqnr.SqnrConfig(),
    sqnr.precision_from_dtypes(
        "float32", "float32", input_dtypes=("float8_e4m3fn", "float8_e4m3fn")
    ),
)  # About 22.5 dB: pass --min_sqnr_db=22.5 to xparity verify.
```

Format (one rounding) | $S$
--------------------- | -------
`float16`             | 73.7 dB
`bfloat16`            | 55.6 dB
`float8_e4m3fn`       | 31.5 dB
`float8_e5m2`         | 25.5 dB
`float4_e2m1fn`       | 16.3 dB
`mxfp8_e4m3`          | 30.7 dB
`mxfp8_e5m2`          | 25.4 dB
`nvfp4`               | 20.4 dB
`mxfp4`               | 18.8 dB

Rounding sources (float32 output) | $E$     | `auto` floor
--------------------------------- | ------- | ------------
fp16 x fp16 operands              | 70.6 dB | 64.6 dB
bf16 x bf16 operands              | 52.6 dB | 46.6 dB
e4m3 x e4m3 operands              | 28.5 dB | 22.5 dB
mxfp8_e4m3 x mxfp8_e4m3 operands  | 27.7 dB | 21.7 dB
mxfp4 weights x bf16 activations  | 18.8 dB | 12.8 dB
none (float32 only)               | n/a     | 45 dB

`auto` counts one rounding into the output dtype. A kernel that rounds to bf16
several times inside (layernorm, softmax, chained bf16 ops) can measure below
the bf16 floor and still be honest: a bf16-ops layernorm measures 44-46 dB and
softmax about 48 dB, against a 49.6 dB floor. For such a kernel, pass an
explicit `--min_sqnr_db` from the kernel spec. With an oracle, a reference of
this kind fails with `reference_below_floor` at the `auto` floor.

`threshold_source` tells where the floor came from: `"explicit"` (a number was
passed, or `six_gate_v1` set its 45 dB floor because the flag was omitted),
`"auto:<derivation>"` (each term is listed as `name:dB`, and the accumulator
term as `acc_<dtype>(K=...,K0=2)`), or `"default"` (the 45 dB fallback of
`auto` when it cannot model the formats, for example with an int8 operand,
output or accumulator dtype). Always quote it with the verdict. If the
derivation does not match the kernel, pass an explicit `--min_sqnr_db` from the
kernel spec. Never lower the floor to make a candidate pass.

## The Oracle Rule

-   No oracle: the leaf fails if `sqnr_db < threshold_db`.
-   With an oracle, the rule has two steps:
    1.  If the reference is below the floor against the oracle
        (`reference_oracle_sqnr_db < threshold_db`), the leaf fails with
        `reference_below_floor`. Fix or pin the reference first.
    2.  Otherwise, the candidate fails only if `oracle_sqnr_db <
        min(threshold_db, reference_oracle_sqnr_db + config.oracle_delta_db)`.
        `config.oracle_delta_db` is the `SqnrConfig` margin, -3 dB by default.

With a good reference, the candidate passes if it meets the floor against the
truth, or if it is no more than 3 dB worse than the reference. Step 1 stops a
candidate from passing because it is close to a broken reference. The rule uses
both terms in step 2 because:

-   The delta alone fails an honest bf16 candidate against a float32 reference.
-   `sqnr_db` alone gives a false red when the reference is lossy. A lossy
    reference caps the measured value: an fp16 candidate (73.7 dB true) measures
    about 55.5 dB against a bf16 reference.

On TPU, a `jnp.dot` reference at `Precision.DEFAULT` rounds its inputs to bf16.
Read `reference_oracle_sqnr_db` before you trust `sqnr_db`, and choose the
oracle with care:

-   An explicit oracle callable is independent of the reference: a host NumPy
    float64 function, or the reference math with
    `precision=jax.lax.Precision.HIGHEST` (Pinned-Float32). Use one of these to
    catch a reference that loses precision on the accelerator.
-   `--kernel_oracle="auto"` re-runs `kernel_ref` with its float inputs
    promoted to float64. It is not Pinned-Float32. JAX needs 64-bit mode for
    it: `jax.config.update("jax_enable_x64", True)`, or `JAX_ENABLE_X64=1` in
    the environment of the CLI. Without that, JAX runs the oracle in float32,
    and an unpinned TPU `jnp.dot` still rounds to bf16, so the oracle has the
    same loss as the reference.

## Read the Report

With any verdict, quote `coverage.regimes_not_run` next to the verdict in your
answer. The `--verdict_only` output has it as `regimes_not_run`. Checking it
only in code is not enough.

Per batch (and per leaf), the `sqnr` dict holds:

-   `passed`, `sqnr_status`, `threshold_db`, `threshold_source`, `gate_math`,
    `failed_reasons` (`"<path>: <metric> <value> < <bound>"`).
-   `warnings`: messages that do not change the verdict (for example, the 45 dB
    fallback for a narrow integer operand, output or accumulator dtype in a
    `PrecisionSpec`).
-   Always gated: `sqnr_db` (no oracle), or the oracle rule (with an oracle)
    on `oracle_sqnr_db` and `reference_oracle_sqnr_db`. This holds under every
    preset, also with `strict=False`. The report field `oracle_delta_db` is the
    measured gap `oracle_sqnr_db - reference_oracle_sqnr_db`, not the
    `SqnrConfig` margin of the same name.
-   Gated under `strict` with `derived` (the default):
    `max_block_noise_ratio_db`, `max_error_ulp`, `min_delta_sqnr_db`, and the
    correlation checks `centered_corr_snr_db`, `diff_corr_snr_db` and
    `plaquette_corr_snr_db`. The correlation checks appear only in
    `failed_reasons`.
-   Gated under `strict` with `six_gate_v1`: `centered_snr_db`,
    `min_block_sqnr_db`, `min_ac_snr_db`, `min_delta_sqnr_db`,
    `peak_error_rms_ratio`, and the centered and macro correlation checks.
-   Report only under `derived`: `min_block_sqnr_db`, `min_ac_snr_db`,
    `centered_snr_db`.
-   Report only under every preset: `cosine_similarity`, `si_snr_db`,
    `best_fit_scale`.
-   `min_delta_sqnr_db` needs the kernel inputs. `validate_kernels`, `verify`
    and `replay` pass them in strict mode for float outputs (inputs with the
    output's element count). `validate_arrays` and `compute_sqnr` have no
    inputs, so they skip the check (it reads None). `audit_leaf` runs it when
    you pass `input_views` with the output's shape. It skips a view of
    another shape and does not reshape it.

The report's `sqnr_summary` merges the leaves under the `sqnr` contract when any
output uses that contract, and all leaves otherwise. It takes the minimum of the
dB and cosine fields and the maximum of `peak_error_rms_ratio`,
`max_block_noise_ratio_db` and `max_error_ulp`.

Under the `sqnr` contract, a complex output is read as (real, imag) pairs. A
`worst_offender` index has a trailing axis of size 2 (index `(..., 1)` is the
imaginary part), and NaN/Inf counts are per part (`complex(nan, nan)` counts 2).

## Why the Strict Checks Exist

Global SQNR averages the error over the whole output. If a kernel zeros $n$ of
$N$ elements (a skipped tile, a dropped expert), global SQNR is at most $10
\log_{10}(N/n)$: 1 tile in 1024 still reads 30 dB. The strict checks catch what
the average hides:

-   **Block noise** (`max_block_noise_ratio_db`): one block with much more noise
    than honest rounding gives.
-   **Correlation** (`centered_corr_snr_db`, `diff_corr_snr_db`,
    `plaquette_corr_snr_db`): constant, mean-only and bias shortcuts. They check
    that the candidate follows the reference on three signals, each with its
    window mean removed: the window itself, the first differences along rows
    and columns (these remove per-row and per-column constants), and the 2D
    mixed second differences (these remove a separable row-plus-column bias).
-   **Input delta** (`min_delta_sqnr_db`): `return x` or `x + small` (near 0
    dB). It runs only when the kernel inputs are available (see Read the
    Report).
-   **Spike** (`max_error_ulp`): one element far outside honest rounding.

Keep `--sqnr_strict=True` for a gate. Use `False` only to measure.

## Regimes for a Narrow-Input Kernel

FP8 inputs are an exception to the default below. Without `regimes` and without
`kernel_oracle`, a `float8_*` input dtype runs only the `normal` regime at every
tier, also `presubmit` and `deep_fuzzing`. To gate an FP8 kernel without an
oracle on the other regimes, pass `regimes` as shown below, or `regimes="all"`
(`--regimes=all`). In your answer, say which regimes ran. Tell the user to quote
`coverage` (`regimes_run` and `regimes_not_run`) with every verdict, also with
the verdict of the rerun with `regimes`. Do not change the floor or the strict
checks to make up for regimes that did not run.

Without `regimes`, `validate_kernels` and `verify` gate on every input regime
of the suite at the `presubmit` and `deep_fuzzing` tiers. With `kernel_oracle`
set, they do so at every tier, also `fast_agent`. A kernel that rounds its
inputs to a narrow float format (FP8, FP4 or bf16) cannot pass two of them:

-   `cancellation`: the inputs are pairs such as 1000 and -999.9. The narrow
    format rounds both to the same magnitude, so the pair cancels and SQNR is
    about 0 dB. The information is lost before the kernel computes.
-   `per_channel_outliers`: a few output columns are about 50 times larger than
    the rest. The block-noise and spike checks use one noise model for the
    whole output, so the small elements in those columns read as too noisy.

For such a kernel, pass
`regimes=("normal", "scale_sweep", "student_t", "outliers", "boundary")`
(on the command line,
`--regimes="normal,scale_sweep,student_t,outliers,boundary"`) and keep the
floor. Never lower the floor to pass the other regimes. The report's `coverage`
lists the regimes that did not run: quote them with the verdict.

The suite seed is fixed (42 by default), so a result repeats from run to run.
With some seeds, an honest narrow float kernel still fails
`max_block_noise_ratio_db` or `max_error_ulp` on one or two `student_t` or
`outliers` batches, while its global SQNR matches the model. This was measured
on bf16, FP8 e4m3 and FP8 e5m2 matmuls. Do not lower the floor or turn off the
strict checks to pass it. Quote the failing batch and reason with the verdict.

This set fits float formats only. An integer kernel (int8, int4) with one scale
per tensor, row or column also misses a typical floor on `student_t` and
`outliers`. One scale must cover the heavy tail, so the small values get few
levels. This is a real accuracy limit of the kernel, not a test artifact. Gate
such a kernel on `regimes=("normal", "scale_sweep", "boundary")`. Then run a
second, report-only call in the same script or check, with
`regimes=("student_t", "outliers")` and `sqnr.SqnrConfig(strict=False)`, and do
not assert on its verdict. Include its `student_t` and `outliers` SQNR in your
answer, next to the verdict. A suggestion in prose is not enough.

## Why Cosine Is Report-Only

Cosine similarity ignores scale. A candidate that returns `0.5 * y` has cosine
1.0 but SQNR 6.0 dB. A cosine of 0.999 can hide a wrong scale factor, a wrong
dequantization scale, or one bad row. `si_snr_db` is also scale-invariant.
`best_fit_scale` is not: it is the least-squares scale from the reference to the
candidate, so `0.5 * y` gives 0.5. Both are report-only. Use them to diagnose a
scale bug (`best_fit_scale` far from 1.0), never as a gate. When someone gives a
cosine number, convert the question to SQNR with this contract and check the
local fields.
