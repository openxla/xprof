# Copyright 2026 The XProf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for eval_jax_lib.

Runs on CPU. The measurement semantics under test -- warmup separation, input
materialization, per-sample trace boundaries, and the shape of the results
dictionary -- are backend-independent, so CPU coverage is meaningful even
though the engine targets TPU.
"""

import glob
import json
import os
from unittest import mock

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
    # A function that no other test uses: `jax.jit` caches compiled code per
    # function, so a function compiled by an earlier test would not compile
    # here.
    def triple(x):
      return x * 3

    result = _profile_once(self.tmpdir, func=triple)
    self.assertLess(result.warmup_sec, result.compilation_sec)

  def test_writes_a_loadable_trace(self):
    result = _profile_once(self.tmpdir)
    traces = glob.glob(
        os.path.join(result.trace_dir, "plugins", "profile", "*", "*.xplane.pb")
    )
    self.assertNotEmpty(traces)

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

  def test_records_the_module_name_from_the_lowered_text(self):
    # The trace reader matches samples by this name, so it must be the name
    # the profiler writes on the `XLA Modules` line.
    result = _profile_once(self.tmpdir)
    self.assertEqual(result.module_name, "jit__double")

  def test_module_name_is_empty_without_a_jit_boundary(self):
    class Unjitted:
      skip_jit = True

      def __call__(self, x):
        return x

    result = _profile_once(self.tmpdir, func=Unjitted())
    self.assertEqual(result.module_name, "")

  def test_module_name_parser_reads_bare_and_quoted_names(self):
    lowered = mock.Mock()
    lowered.as_text.return_value = "#loc = x\nmodule @jit_f attributes {}\n"
    self.assertEqual(eval_jax_lib._module_name_from_lowered(lowered), "jit_f")
    lowered.as_text.return_value = 'module @"jit_<lambda>" {\n'
    self.assertEqual(
        eval_jax_lib._module_name_from_lowered(lowered), "jit_<lambda>"
    )
    lowered.as_text.side_effect = RuntimeError("no text")
    self.assertEqual(eval_jax_lib._module_name_from_lowered(lowered), "")

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
        "trace_dirs",
        "block_sample_counts",
        "skip_jit",
        "compilation_sec",
        "warmup_sec",
        "active_sec",
        "per_sample_sec",
        "module_name",
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

  def test_each_block_writes_its_own_trace(self):
    for role in ("reference", "custom"):
      trace_dirs = self.results[f"{role}_trace_dirs"]
      self.assertEqual(
          trace_dirs,
          [
              os.path.join(self.tmpdir, f"{role}_profile"),
              os.path.join(self.tmpdir, f"{role}_profile_block1"),
          ],
      )
      # Block 0 keeps the directory that a run with one block writes to.
      self.assertEqual(self.results[f"{role}_trace_dir"], trace_dirs[0])
      self.assertEqual(self.results[f"{role}_block_sample_counts"], [1, 1])
      for trace_dir in trace_dirs:
        self.assertNotEmpty(
            glob.glob(
                os.path.join(
                    trace_dir, "plugins", "profile", "*", "*.xplane.pb"
                )
            )
        )

  def test_roles_agree_numerically(self):
    verdicts = eval_jax_lib.check_equal(
        self.results["reference_outputs"], self.results["custom_outputs"]
    )
    self.assertEqual(verdicts, [True, True])


class BlockOrderTest(absltest.TestCase):

  def test_one_role_runs_its_blocks_in_order(self):
    self.assertEqual(
        eval_jax_lib.block_order(["reference"], 2),
        [("reference", 0), ("reference", 1)],
    )

  def test_two_roles_run_a_b_b_a(self):
    self.assertEqual(
        eval_jax_lib.block_order(["reference", "custom"], 2),
        [("reference", 0), ("custom", 0), ("custom", 1), ("reference", 1)],
    )

  def test_three_roles_run_a_b_c_c_b_a(self):
    self.assertEqual(
        [role for role, _ in eval_jax_lib.block_order(["a", "b", "c"], 2)],
        ["a", "b", "c", "c", "b", "a"],
    )

  def test_more_blocks_keep_alternating(self):
    self.assertEqual(
        [role for role, _ in eval_jax_lib.block_order(["a", "b"], 3)],
        ["a", "b", "b", "a", "a", "b"],
    )

  def test_the_blocks_of_a_role_are_in_increasing_order(self):
    order = eval_jax_lib.block_order(["a", "b", "c"], 4)
    for role in ("a", "b", "c"):
      self.assertEqual(
          [block for name, block in order if name == role], [0, 1, 2, 3]
      )


class SampleBlocksTest(absltest.TestCase):

  def test_splits_into_contiguous_blocks_of_near_equal_size(self):
    self.assertEqual(
        eval_jax_lib.sample_blocks(5, 2), [range(0, 2), range(2, 5)]
    )
    self.assertEqual(
        eval_jax_lib.sample_blocks(50, 2), [range(0, 25), range(25, 50)]
    )

  def test_never_makes_more_blocks_than_samples(self):
    self.assertEqual(eval_jax_lib.sample_blocks(1, 2), [range(0, 1)])

  def test_always_makes_one_block(self):
    self.assertEqual(eval_jax_lib.sample_blocks(0, 2), [range(0, 0)])
    self.assertEqual(eval_jax_lib.sample_blocks(3, 0), [range(0, 3)])


class AlternatingBlocksTest(absltest.TestCase):
  """How `profile_funcs` interleaves the timed blocks of the roles."""

  def setUp(self):
    super().setUp()
    self.tmpdir = self.create_tempdir().full_path

  def _run(self, funcs, num_warmups=1, num_samples=4, **kwargs):
    warmup_keys, actual_keys = eval_jax_lib.sample_keys(
        3, num_warmups, num_samples
    )
    results = eval_jax_lib.profile_funcs(
        funcs,
        _indexed_input_generator,
        warmup_keys,
        actual_keys,
        list(range(num_warmups)),
        list(range(num_samples)),
        config_name="default",
        directory=self.tmpdir,
        **kwargs,
    )
    return results, actual_keys

  def test_traces_open_in_the_balanced_order(self):
    with mock.patch.object(
        eval_jax_lib, "trace_context", wraps=eval_jax_lib.trace_context
    ) as traced:
      self._run([("reference", _double), ("custom", lambda x: x + x)])
    self.assertEqual(
        [os.path.basename(call.args[0]) for call in traced.call_args_list],
        [
            "reference_profile",
            "custom_profile",
            "custom_profile_block1",
            "reference_profile_block1",
        ],
    )

  def test_sample_i_of_every_role_sees_the_same_input(self):
    # The pairing that the summarizer relies on: position i of each role's
    # outputs and times is the sample made from key i and index i.
    results, actual_keys = self._run(
        [("reference", _double), ("custom", lambda x: x + x)], num_samples=5
    )
    self.assertEqual(results["reference_block_sample_counts"], [2, 3])
    for role in ("reference", "custom"):
      outputs = results[f"{role}_outputs"]
      self.assertLen(outputs, 5)
      self.assertLen(results[f"{role}_per_sample_sec"], 5)
      for index, (output, key) in enumerate(zip(outputs, actual_keys)):
        (expected_input,), _ = _indexed_input_generator(key, sample_index=index)
        np.testing.assert_allclose(
            np.asarray(output), np.asarray(expected_input) * 2, rtol=1e-6
        )

  def test_a_single_role_runs_in_one_block(self):
    with mock.patch.object(
        eval_jax_lib, "trace_context", wraps=eval_jax_lib.trace_context
    ) as traced:
      results, _ = self._run([("reference", _double)])
    self.assertEqual(traced.call_count, 1)
    self.assertEqual(results["reference_block_sample_counts"], [4])
    self.assertEqual(
        results["reference_trace_dirs"], [results["reference_trace_dir"]]
    )

  def test_one_block_runs_each_role_once(self):
    results, _ = self._run(
        [("reference", _double), ("custom", lambda x: x + x)], num_blocks=1
    )
    for role in ("reference", "custom"):
      self.assertEqual(results[f"{role}_block_sample_counts"], [4])

  def test_each_later_block_is_rewarmed(self):
    class Counting:
      skip_jit = True

      def __init__(self):
        self.calls = 0

      def __call__(self, x):
        self.calls += 1
        return x

    reference, custom = Counting(), Counting()
    self._run(
        [("reference", reference), ("custom", custom)],
        num_warmups=3,
        num_samples=4,
    )
    # 1 compile-timed call + 3 warmups + 2 rewarm calls before block 1 + 4
    # timed samples.
    self.assertEqual(reference.calls, 10)
    self.assertEqual(custom.calls, 10)

  def test_rejects_fewer_than_one_block(self):
    with self.assertRaises(ValueError):
      self._run([("reference", _double)], num_blocks=0)


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


class DescribeEnvironmentTest(absltest.TestCase):

  def test_records_the_devices_and_software(self):
    env = eval_jax_lib.describe_environment()
    self.assertEqual(env["platform"], jax.devices()[0].platform)
    self.assertLen(jax.devices(), env["device_count"])
    self.assertNotEmpty(env["device_kind"])
    self.assertNotEmpty(env["python_version"])
    self.assertNotEmpty(env["jax_version"])

  def test_flags_come_from_the_environment(self):
    with mock.patch.dict(
        os.environ,
        {"XLA_FLAGS": "--xla_a=1 --xla_b=2", "LIBTPU_INIT_ARGS": "--xla_c"},
    ):
      env = eval_jax_lib.describe_environment()
    self.assertEqual(env["xla_flags"], ["--xla_a=1", "--xla_b=2", "--xla_c"])

  def test_keys_match_the_proto(self):
    # Keys the summarizer does not recognize are dropped, so a typo here would
    # silently lose a field.
    env = eval_jax_lib.describe_environment()
    allowed = {
        "host_name",
        "python_version",
        "jax_version",
        "jaxlib_version",
        "libtpu_version",
        "xprof_version",
        "platform",
        "platform_version",
        "device_kind",
        "device_count",
        "accelerator_type",
        "topology",
        "xla_flags",
    }
    self.assertContainsSubset(set(env), allowed)


if __name__ == "__main__":
  jax.config.update("jax_platforms", "cpu")
  absltest.main()
