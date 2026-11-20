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

```bash
xparity verify \
  --kernel_ref="my_pkg.kernels:ref_fn" \
  --kernel_candidate="my_pkg.kernels:int8_matmul_fn" \
  --shapes="[(256, 4096)]" \
  --dtype_str="float32" \
  --contract="sqnr" \
  --min_sqnr_db=30 \
  --kernel_oracle="auto"
```

Flag            | Default   | Meaning
--------------- | --------- | -------
`--min_sqnr_db` | `auto`    | Global floor in dB, or `auto` (see below). Integer formats need an explicit value.
`--sqnr_alpha`  | `1e-6`    | False-alarm rate per output on honest noise that the noise model fits. The strict check bounds come from it. Two summation orders of one bf16 matmul fail about 1 leaf in 120 (65536 elements) on `max_error_ulp`.
`--sqnr_preset` | `derived` | `derived`, or `six_gate_v1` (fixed constants, 45 dB default floor).
`--sqnr_strict` | `True`    | `False` runs only the global floor and the oracle rule.

```python
from xprof.xparity import CONTRACT_SQNR
from xprof.xparity import sqnr
from xprof.xparity import validate_kernels

report = validate_kernels(
    kernel_ref=reference_fn,
    kernel_candidate=int8_matmul_fn,
    shapes=[(256, 4096)],
    dtype_str="float32",
    contract=CONTRACT_SQNR,
    kernel_oracle="auto",
    sqnr_config=sqnr.SqnrConfig(min_sqnr_db=30.0),  # None means SqnrConfig().
)

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
```

`sqnr.SqnrConfig.from_options(min_sqnr_db=..., alpha=..., strict=...,
preset=...)` builds the same config as the CLI flags.
`sqnr.SqnrConfig.six_gate_v1()` selects the fixed-constant preset.

## How `auto` Sets the Floor

`sqnr.expected_sqnr_db(precision)` adds the noise power of each rounding source
$i$ with per-rounding SQNR $S_i$:

$$E = -10 \log_{10} \sum_i 10^{-S_i/10}$$

-   The sources are each operand dtype, the output dtype, and the accumulator
    dtype. A source counts only if it is narrower than float32.
-   `PrecisionSpec.output_dtype` defaults to the reference dtype. A bf16 output
    adds a bf16 term: bf16 x bf16 operands with a bf16 output give 50.8 dB, not
    52.6 dB.
-   For a plain float format, $S = 6.02\,p + 7.44$ dB. $p$ is the significand
    width including the hidden bit (bf16 8, fp16 11, e4m3 4, e5m2 3).
-   For block-scaled formats and `float4_e2m1fn`, $S$ is a simulated value
    (table below). Pass the format name (for example `"mxfp4"`) in
    `PrecisionSpec.operand_dtypes`.
-   An accumulator narrower than float32 with reduction length $K > K_0$ adds
    the source $S_{acc} - 10 \log_{10}(K/K_0)$, with $K_0 = 2$. A bf16
    accumulator with $K = 4096$ is a 22.5 dB source by itself.
-   The floor is $E - 6$ dB (`auto_margin_db`).
-   With no source narrower than float32, the floor is 45 dB.
-   `auto` does not model integer formats (int8, int4). For an integer kernel,
    the floor falls back to 45 dB and the report adds a `warnings` entry. Always
    pass an explicit `--min_sqnr_db` from the kernel spec for integer
    quantization.

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
passed), `"auto:<derivation>"` (each term is listed as `name:dB`, and the
accumulator term as `acc_<dtype>(K=...,K0=2)`), or `"default"` (the 45 dB
fallback). Always quote it with the verdict. If the derivation does not match
the kernel, pass an explicit `--min_sqnr_db` from the kernel spec. Never lower
the floor to make a candidate pass.

## The Oracle Rule

-   No oracle: the leaf fails if `sqnr_db < threshold_db`.
-   With an oracle, the rule has two steps:
    1.  If the reference is below the floor against the oracle
        (`reference_oracle_sqnr_db < threshold_db`), the leaf fails with
        `reference_below_floor`. Fix or pin the reference first.
    2.  Otherwise, the candidate fails only if `oracle_sqnr_db <
        min(threshold_db, reference_oracle_sqnr_db + oracle_delta_db)`.
        `oracle_delta_db` is -3 dB by default.

With a good reference, the candidate passes if it meets the floor against the
truth, or if it is no more than 3 dB worse than the reference. Step 1 stops a
candidate from passing because it is close to a broken reference. The rule uses
both terms in step 2 because:

-   The delta alone fails an honest bf16 candidate against a float32 reference.
-   `sqnr_db` alone gives a false red when the reference is lossy. A lossy
    reference caps the measured value: an fp16 candidate (73.7 dB true) measures
    about 55.5 dB against a bf16 reference.

On TPU, a `jnp.dot` reference at `Precision.DEFAULT` rounds its inputs to bf16.
Use `--kernel_oracle="auto"` (Float64 or Pinned-Float32) and read
`reference_oracle_sqnr_db` before you trust `sqnr_db`.

## Read the Report

Per batch (and per leaf), the `sqnr` dict holds:

-   `passed`, `sqnr_status`, `threshold_db`, `threshold_source`, `gate_math`,
    `failed_reasons` (`"<path>: <metric> <value> < <bound>"`).
-   `warnings`: messages that do not change the verdict (for example, the 45 dB
    fallback for an integer format).
-   Always gated: `sqnr_db` (no oracle), or the oracle rule on `oracle_sqnr_db`
    and `reference_oracle_sqnr_db`.
-   Gated under `strict` with `derived` (the default):
    `max_block_noise_ratio_db`, `max_error_ulp`, `min_delta_sqnr_db`, and the
    correlation checks `centered_corr_snr_db`, `diff_corr_snr_db` and
    `plaquette_corr_snr_db`. The correlation checks appear only in
    `failed_reasons`.
-   Gated under `strict` with `six_gate_v1`: `centered_snr_db`,
    `min_block_sqnr_db`, `min_ac_snr_db`, `min_delta_sqnr_db`,
    `peak_error_rms_ratio`, and the centered and macro correlation checks.
-   Report only under `derived`: `min_block_sqnr_db`, `min_ac_snr_db`,
    `centered_snr_db`, `oracle_delta_db`.
-   Report only under every preset: `cosine_similarity`, `si_snr_db`,
    `best_fit_scale`.

The report's `sqnr_summary` merges all leaves. It takes the minimum of the dB
and cosine fields and the maximum of `peak_error_rms_ratio`,
`max_block_noise_ratio_db` and `max_error_ulp`.

## Why the Strict Checks Exist

Global SQNR averages the error over the whole output. If a kernel zeros $n$ of
$N$ elements (a skipped tile, a dropped expert), global SQNR is at most $10
\log_{10}(N/n)$: 1 tile in 1024 still reads 30 dB. The strict checks catch what
the average hides:

-   **Block noise** (`max_block_noise_ratio_db`): one block with much more noise
    than honest rounding gives.
-   **Correlation** (`centered_corr_snr_db`, `diff_corr_snr_db`,
    `plaquette_corr_snr_db`): constant, mean-only and bias shortcuts. They check
    that the candidate follows the reference after the mean, the first
    differences and the 2D second differences are removed.
-   **Input delta** (`min_delta_sqnr_db`): `return x` or `x + small` (near 0
    dB).
-   **Spike** (`max_error_ulp`): one element far outside honest rounding.

Keep `--sqnr_strict=True` for a gate. Use `False` only to measure.

## Why Cosine Is Report-Only

Cosine similarity ignores scale. A candidate that returns `0.5 * y` has cosine
1.0 but SQNR 6.0 dB. A cosine of 0.999 can hide a wrong scale factor, a wrong
dequantization scale, or one bad row. `si_snr_db` and `best_fit_scale` are also
scale-invariant. Use them to diagnose a scale bug (`best_fit_scale` far from
1.0), never as a gate. When someone gives a cosine number, convert the question
to SQNR with this contract and check the local fields.
