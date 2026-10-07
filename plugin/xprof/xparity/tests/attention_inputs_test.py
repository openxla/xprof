"""Tests for the attention input generator."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator


def _attention(
    q, k, v, mask=None, q_segment_ids=None, kv_segment_ids=None, stable=True
):
  """Reference attention in float32 with GQA, masks and packed segments."""
  q, k, v = (np.asarray(t, dtype=np.float32) for t in (q, k, v))
  groups = q.shape[1] // k.shape[1]
  k = np.repeat(k, groups, axis=1)
  v = np.repeat(v, groups, axis=1)
  scale = np.float32(1.0) / np.sqrt(np.float32(q.shape[-1]))
  logits = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
  allowed = np.ones(logits.shape, dtype=bool)
  if mask is not None:
    allowed &= np.asarray(mask, dtype=bool)
  if q_segment_ids is not None:
    assert kv_segment_ids is not None
    allowed &= (
        q_segment_ids[:, None, :, None] == kv_segment_ids[:, None, None, :]
    )
  logits = np.where(allowed, logits, -np.inf)
  if stable:
    logits = logits - logits.max(axis=-1, keepdims=True)
  with np.errstate(over="ignore", invalid="ignore"):
    weights = np.exp(logits)
    weights = weights / weights.sum(axis=-1, keepdims=True)
  return np.einsum("bhqk,bhkd->bhqd", weights, v)


def _naive_attention(q, k, v, **kwargs):
  return _attention(q, k, v, stable=False, **kwargs)


def _by_regime(suite):
  return {b["regime"]: b for b in suite}


class GenerateAttentionSuiteTest(parameterized.TestCase):

  def test_shapes_and_gqa(self):
    suite = numerical_generator.generate_attention_suite(
        batch=2, num_heads=8, q_len=16, kv_len=16, head_dim=32, num_kv_heads=2
    )
    self.assertEqual(
        [b["regime"] for b in suite],
        list(numerical_generator.ATTENTION_REGIMES),
    )
    for b in suite:
      q, k, v = b["args"]
      self.assertEqual(q.shape, (2, 8, 16, 32))
      self.assertEqual(k.shape, (2, 2, 16, 32))
      self.assertEqual(v.shape, (2, 2, 16, 32))

  def test_large_logits_reach_exp_overflow(self):
    batch = _by_regime(
        numerical_generator.generate_attention_suite(
            1, 1, 8, 8, 64, dtype_str="float32"
        )
    )["attention_large_logits"]
    q, k, _ = (a.astype(np.float32) for a in batch["args"])
    logits = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(np.float32(64))
    self.assertGreater(float(logits.max()), 88.0)

  def test_masks_never_empty_a_query_row(self):
    by_regime = _by_regime(
        numerical_generator.generate_attention_suite(4, 2, 12, 12, 8)
    )
    for regime in ("attention_causal", "attention_padding"):
      mask = by_regime[regime]["kwargs"]["mask"]
      self.assertEqual(mask.shape, (4, 1, 12, 12))
      self.assertTrue(np.all(mask.any(axis=-1)), regime)
    segments = by_regime["attention_segments"]["kwargs"]
    q_ids = segments["q_segment_ids"]
    self.assertEqual(q_ids.shape, (4, 12))
    self.assertTrue(np.all(np.diff(q_ids, axis=-1) >= 0))
    np.testing.assert_array_equal(q_ids, segments["kv_segment_ids"])

  def test_segments_skipped_for_cross_attention(self):
    suite = numerical_generator.generate_attention_suite(1, 2, 8, 16, 8)
    self.assertNotIn("attention_segments", _by_regime(suite))
    for b in suite:
      if "mask" in b["kwargs"]:
        self.assertEqual(b["kwargs"]["mask"].shape, (1, 1, 8, 16))

  @parameterized.named_parameters(
      ("bad_size", dict(head_dim=0), "must be positive"),
      ("bad_gqa", dict(num_kv_heads=3), "must divide"),
      ("bad_regime", dict(regimes=("attention_sparse",)), "Unknown attention"),
  )
  def test_invalid_arguments(self, overrides, message):
    kwargs = dict(batch=1, num_heads=4, q_len=4, kv_len=4, head_dim=8)
    kwargs.update(overrides)
    with self.assertRaisesRegex(ValueError, message):
      numerical_generator.generate_attention_suite(**kwargs)


class AttentionValidationTest(absltest.TestCase):

  def _suite(self):
    return numerical_generator.generate_attention_suite(
        batch=2,
        num_heads=4,
        q_len=16,
        kv_len=16,
        head_dim=32,
        dtype_str="float32",
        num_kv_heads=2,
    )

  def test_reference_matches_itself_on_every_regime(self):
    report = numerical_validator.validate_kernels(
        _attention,
        _attention,
        shapes=(1,),
        dtype_str="float32",
        test_suite=self._suite(),
    )
    self.assertTrue(report.is_numerically_equivalent, report.summary_message)
    self.assertEmpty(report.coverage["regimes_not_run"])

  def test_softmax_without_max_subtraction_fails_large_logits(self):
    report = numerical_validator.validate_kernels(
        _attention,
        _naive_attention,
        shapes=(1,),
        dtype_str="float32",
        test_suite=self._suite(),
    )
    self.assertFalse(report.is_numerically_equivalent)
    large = [
        b for b in report.batch_results if b.regime == "attention_large_logits"
    ]
    self.assertLen(large, 1)
    self.assertFalse(large[0].passed)
    self.assertTrue(large[0].has_nan_or_inf)


if __name__ == "__main__":
  absltest.main()
