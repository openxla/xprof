"""Tests for eval_jax_lib.

Runs on CPU. The measurement semantics under test -- warmup separation, input
materialization, per-sample trace boundaries, and the shape of the results
dictionary -- are backend-independent, so CPU coverage is meaningful even
though the engine targets TPU.
"""

import glob
import json
import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from xprof.xloop import eval_jax_lib


def _input_generator(key):
  return (jax.random.normal(key, (8, 8), jnp.float32),), {}


def _indexed_input_generator(key, sample_index=0):
  return (jax.random.normal(key, (8, 8), jnp.float32) + sample_index,), {}


def _double(x):
  return x * 2


def _profile_once(tmpdir, func=_double, generator=_input_generator, **kwargs):
  warmup_keys, actual_keys = eval_jax_lib.sample_keys(0, 1, 2)
  return eval_jax_lib.profile(
      func,
      generator,
      warmup_keys,
      actual_keys,
      list(range(len(warmup_keys))),
      list(range(len(actual_keys))),
      os.path.join(tmpdir, "profile"),
      config_name="default",
      **kwargs,
  )


class SampleKeysTest(absltest.TestCase):

  def test_splits_into_requested_counts(self):
    warmup, actual = eval_jax_lib.sample_keys(0, 2, 3)
    self.assertLen(warmup, 2)
    self.assertLen(actual, 3)

  def test_is_deterministic(self):
    # Every implementation must see byte-identical inputs, or the correctness
    # comparison between them is meaningless.
    first = eval_jax_lib.sample_keys(7, 1, 1)
    second = eval_jax_lib.sample_keys(7, 1, 1)
    for a, b in zip(first[1], second[1]):
      np.testing.assert_array_equal(
          jax.random.key_data(a), jax.random.key_data(b)
      )

  def test_warmup_and_actual_keys_differ(self):
    warmup, actual = eval_jax_lib.sample_keys(0, 1, 1)
    self.assertFalse(
        np.array_equal(
            jax.random.key_data(warmup[0]), jax.random.key_data(actual[0])
        )
    )

  def test_supports_zero_warmups(self):
    warmup, actual = eval_jax_lib.sample_keys(0, 0, 2)
    self.assertEmpty(warmup)
    self.assertLen(actual, 2)


class StaticArgnamesTest(absltest.TestCase):

  def test_absent_attribute_yields_empty(self):
    self.assertEmpty(eval_jax_lib.get_static_argnames(_double))

  def test_sequence_is_returned_as_tuple(self):
    class Marked:
      static_argnames = ["a", "b"]

      def __call__(self, x):
        return x

    self.assertEqual(eval_jax_lib.get_static_argnames(Marked()), ("a", "b"))

  def test_bare_string_is_wrapped(self):
    # A lone string would otherwise iterate into characters.
    class Marked:
      static_argnames = "mode"

      def __call__(self, x):
        return x

    self.assertEqual(eval_jax_lib.get_static_argnames(Marked()), ("mode",))


class ProfileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = self.create_tempdir().full_path

  def test_returns_one_output_per_timed_sample(self):
    result = _profile_once(self.tmpdir)
    self.assertLen(result.outputs, 2)

  def test_outputs_are_correct(self):
    result = _profile_once(self.tmpdir)
    warmup_keys, actual_keys = eval_jax_lib.sample_keys(0, 1, 2)
    del warmup_keys
    for output, key in zip(result.outputs, actual_keys):
      (expected_input,), _ = _input_generator(key)
      np.testing.assert_allclose(
          np.asarray(output), np.asarray(expected_input) * 2, rtol=1e-6
      )

  def test_timings_are_populated(self):
    result = _profile_once(self.tmpdir)
    self.assertGreater(result.compilation_sec, 0.0)
    self.assertGreaterEqual(result.warmup_sec, 0.0)
    self.assertGreater(result.active_sec, 0.0)

  def test_compilation_is_not_charged_to_warmup(self):
    # Compilation is timed off a separate leading call, so warmup reflects
    # steady-state cost. On a trivial kernel warmup should be far cheaper.
    result = _profile_once(self.tmpdir)
    self.assertLess(result.warmup_sec, result.compilation_sec)

  def test_writes_a_loadable_trace(self):
    result = _profile_once(self.tmpdir)
    traces = glob.glob(
        os.path.join(result.trace_dir, "plugins", "profile", "*", "*.xplane.pb")
    )
    self.assertNotEmpty(traces)

  def test_session_ids_are_absent(self):
    # This backend writes traces to disk rather than uploading them.
    result = _profile_once(self.tmpdir)
    self.assertIsNone(result.session_id)
    self.assertIsNone(result.isolated_session_id)
    self.assertEmpty(result.isolated_trace_dir)

  def test_generator_receives_sample_index_when_accepted(self):
    result = _profile_once(self.tmpdir, generator=_indexed_input_generator)
    warmup_keys, actual_keys = eval_jax_lib.sample_keys(0, 1, 2)
    del warmup_keys
    for i, (output, key) in enumerate(zip(result.outputs, actual_keys)):
      (expected_input,), _ = _indexed_input_generator(key, sample_index=i)
      np.testing.assert_allclose(
          np.asarray(output), np.asarray(expected_input) * 2, rtol=1e-6
      )

  def test_config_name_is_forwarded_when_accepted(self):
    seen = []

    def func(x, config_name=""):
      seen.append(config_name)
      return x

    _profile_once(self.tmpdir, func=func)
    self.assertEqual(set(seen), {"default"})

  def test_static_argnames_are_honored(self):
    class Scaled:
      static_argnames = ("scale",)

      def __call__(self, x, scale=1):
        return x * scale

    def generator(key):
      return (jax.random.normal(key, (4,), jnp.float32),), {"scale": 3}

    result = _profile_once(self.tmpdir, func=Scaled(), generator=generator)
    self.assertLen(result.outputs, 2)

  def test_skip_jit_is_respected(self):
    class Counting:
      skip_jit = True

      def __init__(self):
        self.calls = 0

      def __call__(self, x):
        self.calls += 1
        return x

    func = Counting()
    _profile_once(self.tmpdir, func=func)
    # Without a jit boundary the Python body runs on every invocation:
    # 1 compile-timed call + 1 warmup + 2 timed.
    self.assertEqual(func.calls, 4)

  def test_zero_warmups_still_measures(self):
    _, actual_keys = eval_jax_lib.sample_keys(0, 0, 2)
    result = eval_jax_lib.profile(
        _double,
        _input_generator,
        [],
        actual_keys,
        [],
        list(range(len(actual_keys))),
        os.path.join(self.tmpdir, "p"),
        config_name="default",
    )
    self.assertLen(result.outputs, 2)
    # Not exactly zero: timing an empty loop still costs a perf_counter pair.
    self.assertLess(result.warmup_sec, 1e-3)


class ProfileFuncsTest(absltest.TestCase):
  """The results dictionary keys are pinned as the summarizer contract."""

  def setUp(self):
    super().setUp()
    self.tmpdir = self.create_tempdir().full_path
    warmup_keys, actual_keys = eval_jax_lib.sample_keys(0, 1, 2)
    self.results = eval_jax_lib.profile_funcs(
        [("reference", _double), ("custom", lambda x: x + x)],
        _input_generator,
        warmup_keys,
        actual_keys,
        [0],
        [0, 1],
        config_name="default",
        directory=self.tmpdir,
    )

  def test_emits_the_expected_keys_for_each_role(self):
    suffixes = (
        "outputs",
        "trace_dir",
        "xprof_session_id",
        "isolated_trace_dir",
        "isolated_xprof_session_id",
        "skip_jit",
        "compilation_sec",
        "warmup_sec",
        "active_sec",
        "per_sample_sec",
    )
    expected = set()
    for role in ("reference", "custom"):
      for suffix in suffixes:
        expected.add(f"{role}_{suffix}")
    self.assertEqual(set(self.results), expected)

  def test_per_sample_times_are_one_per_sample_and_positive(self):
    # The summarizer sizes its noise threshold from the spread of these, so a
    # missing or short list silently degrades it to a fixed floor.
    for role in ("reference", "custom"):
      times = self.results[f"{role}_per_sample_sec"]
      self.assertLen(times, len(self.results[f"{role}_outputs"]))
      self.assertTrue(all(t > 0 for t in times), f"{role}: {times}")

  def test_traces_are_written_per_role(self):
    self.assertNotEqual(
        self.results["reference_trace_dir"], self.results["custom_trace_dir"]
    )
    for role in ("reference", "custom"):
      self.assertTrue(os.path.isdir(self.results[f"{role}_trace_dir"]))

  def test_roles_agree_numerically(self):
    verdicts = eval_jax_lib.check_equal(
        self.results["reference_outputs"], self.results["custom_outputs"]
    )
    self.assertEqual(verdicts, [True, True])


class CheckEqualTest(absltest.TestCase):

  def test_identical_outputs_match(self):
    a = [jnp.ones((4,)), jnp.zeros((4,))]
    self.assertEqual(eval_jax_lib.check_equal(a, list(a)), [True, True])

  def test_divergent_output_is_flagged(self):
    a = [jnp.ones((4,)), jnp.ones((4,))]
    b = [jnp.ones((4,)), jnp.ones((4,)) * 2]
    self.assertEqual(eval_jax_lib.check_equal(a, b), [True, False])

  def test_tolerance_is_applied(self):
    a = [jnp.ones((4,))]
    b = [jnp.ones((4,)) + 1e-6]
    self.assertEqual(
        eval_jax_lib.check_equal(a, b, rtol=1e-3, atol=1e-3), [True]
    )
    self.assertEqual(
        eval_jax_lib.check_equal(a, b, rtol=0.0, atol=0.0), [False]
    )

  def test_length_mismatch_returns_false(self):
    self.assertFalse(
        eval_jax_lib.check_equal([jnp.ones((4,))], [jnp.ones((4,))] * 2)
    )

  def test_none_returns_false(self):
    self.assertFalse(eval_jax_lib.check_equal(None, [jnp.ones((4,))]))
    self.assertFalse(eval_jax_lib.check_equal([jnp.ones((4,))], None))

  def test_empty_returns_false(self):
    self.assertFalse(eval_jax_lib.check_equal([], []))

  def test_shape_mismatch_is_flagged_not_raised(self):
    self.assertEqual(
        eval_jax_lib.check_equal([jnp.ones((4,))], [jnp.ones((5,))]), [False]
    )

  def test_structure_mismatch_is_flagged_not_raised(self):
    self.assertEqual(
        eval_jax_lib.check_equal(
            [(jnp.ones((4,)), jnp.ones((4,)))], [jnp.ones((4,))]
        ),
        [False],
    )

  def test_pytree_outputs_are_compared_leafwise(self):
    a = [{"y": jnp.ones((2,)), "z": jnp.zeros((2,))}]
    b = [{"y": jnp.ones((2,)), "z": jnp.ones((2,))}]
    self.assertEqual(eval_jax_lib.check_equal(a, b), [False])

  def test_nan_in_both_compares_equal(self):
    a = [jnp.array([jnp.nan, 1.0])]
    self.assertEqual(eval_jax_lib.check_equal(a, list(a)), [True])


class BenchmarkFuncForModelTest(absltest.TestCase):

  def test_plain_callable_is_returned_unchanged(self):
    self.assertIs(
        eval_jax_lib.benchmark_func_for_model(_double, _input_generator),
        _double,
    )

  def test_plain_class_is_instantiated_from_init_inputs(self):
    class Scaler:

      def __init__(self, factor):
        self.factor = factor

      def __call__(self, x):
        return x * self.factor

    func = eval_jax_lib.benchmark_func_for_model(
        Scaler, _input_generator, get_init_inputs=lambda: (3,)
    )
    np.testing.assert_allclose(np.asarray(func(jnp.ones((2,)))), [3.0, 3.0])


class SerializeResultsTest(absltest.TestCase):

  def test_writes_results_json_without_output_arrays(self):
    tmpdir = self.create_tempdir().full_path
    eval_jax_lib.serialize_results(
        {
            "reference_outputs": [jnp.ones((4,))],
            "custom_outputs": [jnp.ones((4,))],
            "reference_active_sec": 1.25,
        },
        tmpdir,
    )
    with open(os.path.join(tmpdir, "results.json")) as f:
      written = json.load(f)
    # Output arrays are dropped: they are large, unserializable, and only ever
    # needed inside the worker for the correctness check.
    self.assertEqual(written, {"reference_active_sec": 1.25})

  def test_creates_missing_directory(self):
    target = os.path.join(self.create_tempdir().full_path, "nested")
    eval_jax_lib.serialize_results({"a": 1}, target)
    self.assertTrue(os.path.exists(os.path.join(target, "results.json")))


if __name__ == "__main__":
  jax.config.update("jax_platforms", "cpu")
  absltest.main()
