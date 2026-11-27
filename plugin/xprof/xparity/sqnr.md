# xparity SQNR checks

`sqnr.py` measures the signal-to-quantization-noise ratio (SQNR) between a
reference output and a candidate output, and gates it. It also runs local
checks that catch errors a global SQNR hides: a skipped tile, a constant or
mean-only output, a passthrough of an input, or a single spike. The module
docstring and the docstrings of `audit_leaf`, `SqnrConfig` and
`PrecisionSpec` are the reference. This page lists how to set the floor and
the known limits of the checks.

## Floor

`SqnrConfig(min_sqnr_db="auto")` (the default) derives the floor from a
`PrecisionSpec`: the expected SQNR of honest rounding, minus
`auto_margin_db` (6 dB by default).

*   Each rounding into a format narrower than float32 is one source. An
    operand, the output and the accumulator can each be a source. Noise
    powers add.
*   A narrow accumulator with reduction length K adds a term that falls
    3 dB for each doubling of K.
*   With no source, the floor is 45 dB.
*   Integer formats narrower than 32 bits (int16, int8, int4, int2 and the
    unsigned forms) are not modeled. The floor is 45 dB, with a warning. A
    leaf that goes to the discrete rule does not use the floor and gets no
    warning.
*   AUTO raises `ValueError` when the derived floor is 0 dB or lower, for
    example a long reduction with a bfloat16 accumulator. Pass an explicit
    `min_sqnr_db` in that case.
*   AUTO counts one rounding into the output dtype. Kernels with several
    bfloat16 roundings (layernorm 44-46 dB, softmax about 48 dB) need an
    explicit floor.

`PrecisionSpec` takes format names. Use the canonical NumPy or ml_dtypes
name of a float, complex, integer or bool dtype (`"bfloat16"`,
`"float8_e4m3fn"`, `"int8"`), or a name in `BLOCK_SCALED_FORMATS` (any
case). Other names raise `ValueError`. This includes short forms such as
`"bf16"` and NumPy aliases such as `"i8"` (int64 in NumPy). Use
`precision_from_dtypes` to build a spec from dtype objects.

## Known limits

*   `alpha` holds for noise that the fitted model describes. Two float32
    summation orders of one bfloat16 matmul gave 0 false failures in 620
    trials (K up to 4096, outputs up to 1048576 elements). The 95% upper
    bound is about 1 in 200 leaves.
*   Heavy-tailed matmul inputs can make the spike check (`max_error_ulp`)
    fail an honest bfloat16 matmul with float32 accumulation. The fitted
    noise model follows the output magnitude, but the rounding error of a
    matmul output follows the norms of its input row and column. An output
    that is small by cancellation, in a row with a large input, can get an
    error many times its modeled standard deviation. Examples are 1% or 0.1%
    of the elements of one input scaled by 50, or a few tokens and channels
    scaled by 100 to 1000 (massive activations). At a scale of 30, and in
    some cases at 100 or 1000, the block-noise check fails instead or as
    well. The global SQNR matches the model in these cases. A per-row floor
    on the spike bound removes the spike-check failures, but it also passes
    spikes on small outputs of element-wise casts, so the checks do not use
    one. With `strict=False`, only the global (or oracle) SQNR rule and the
    leaf rules run: these cases pass, and so do most local errors.
*   The correlation checks do not test AC below 16 eps^2 of the energy of
    the reference values in a window (about 4 eps standard deviation). A
    row-bias-only error at that level passes them: 130 dB or more in
    float32, or 34-45 dB on a bfloat16 reference with an explicit 20 dB
    floor. With the AUTO floor, the global SQNR check catches the bfloat16
    case.
*   Windows with AC below about 126 dB (float32), 301 dB (float64) or 30 dB
    (bfloat16) under the mean reference power are not tested. On a bfloat16
    reference with an explicit 20 dB floor, a candidate that adds noise in
    one half and flattens a quiet region at 1-2% of the leaf RMS passes. The
    bfloat16 AUTO floor catches it.
*   The correlation gate uses the uncapped u^2 / 12 of a float8 output, so a
    row below half the smallest step is not graded for AC. On such a row,
    errors up to about 4 subnormal steps pass.
*   The window correlation gate keeps 0.707 times the summed window
    variance, which includes the fitted B. A float32 row with AC below half
    a bfloat16 step, which an honest bfloat16 cast makes flat, can fail
    `centered_corr_snr_db`.
*   A spike at a reference zero goes to the block-noise check only. In a
    bfloat16 ReLU leaf with 1e-3 noise, a 0.02 error at a zero passes and a
    0.05 error fails.
*   The window correlation admission uses the reference only, except that a
    window whose error energy is above sqrt(2) times its AC energy is tested
    only for a significantly negative centered correlation. So noise that
    replaces a small part of the signal (the AC of a DC-dominated leaf, or a
    quiet complex part) passes when the block-noise check and the global
    floor pass. On 1000 + N(0, 1) (64 x 128, float64), noise of 0.8, 1.0 and
    1.5 in place of the AC passes at 57.9, 57.0 and 54.9 dB.
*   Honest float64 noise of about 1e-17 on a block of exact reference zeros
    fails the block-noise check. A complex128 leaf with no fit data (all
    other finite values exact) also fails it.
*   The checks know only the formats in the `PrecisionSpec`. A bfloat16
    matmul audited with a float32 output format only flushes float32
    subnormal inputs to 0, so outputs near 4e-41 become 0. Such leaves pass
    only through the fitted B. With the input formats in the spec, they pass
    with B = 0.
