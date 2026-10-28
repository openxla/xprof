"""End-to-end tests for xloop_lib and xloop_cli.

Runs real JAX kernels through the worker on CPU. These are the tests that
would catch a break in the contract between the item proto, the worker's
module loading, and the harness.
"""

import json
import os

from absl.testing import absltest

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import input_utils
from xprof.xloop import summarize_results_lib
from xprof.xloop import xloop_cli
from xprof.xloop import xloop_lib

_SUM_OF_SQUARES = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    return jnp.sum(x * x)

  return f
"""

# Same value, different formulation: exercises the correctness check passing
# on code that is not textually identical.
_SUM_OF_SQUARES_VIA_DOT = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    return jnp.dot(x, x)

  return f
"""

_WRONG_ANSWER = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    return jnp.sum(x) * 3.0

  return f
"""

_RAISES_ON_IMPORT = """
raise RuntimeError("kernel is broken")


def get_model_identifier():
  return None
"""

_NO_ENTRY_POINT = """
import jax.numpy as jnp
"""

_INPUTS = """
import jax
import jax.numpy as jnp


def get_input_generator():
  def generate(key):
    return (jax.random.normal(key, (32,), jnp.float32),), {}

  return generate
"""

# Uses an auxiliary module, to check that they are written and importable.
_INPUTS_VIA_AUX = """
import jax
import jax.numpy as jnp

from xloop_test_aux import shapes


def get_input_generator():
  def generate(key):
    return (jax.random.normal(key, shapes.SHAPE, jnp.float32),), {}

  return generate
"""

_AUX_SHAPES = """
SHAPE = (8,)
"""


def _item(
    name: str,
    reference: str = _SUM_OF_SQUARES,
    custom: str = _SUM_OF_SQUARES_VIA_DOT,
    configs: dict[str, str] | None = None,
) -> xloop_item_pb2.XLoopItem:
  """Builds a runnable item."""
  item = xloop_item_pb2.XLoopItem(name=name, id=name)
  item.reference_code = reference
  item.custom_code = custom
  for config_name, code in (configs or {"default": _INPUTS}).items():
    item.input_generation_configs[config_name].input_generation_code = code
  return item


class ChipCountFromTopologyTest(absltest.TestCase):

  def test_empty_topology_is_unspecified(self):
    self.assertIsNone(xloop_lib._chip_count_from_topology(""))

  def test_platform_prefix_is_stripped(self):
    self.assertEqual(xloop_lib._chip_count_from_topology("vlp=2x4"), 8)

  def test_bare_shape_is_accepted(self):
    self.assertEqual(xloop_lib._chip_count_from_topology("2x2x1"), 4)

  def test_garbage_is_unspecified(self):
    self.assertIsNone(xloop_lib._chip_count_from_topology("not-a-topology"))


class RunExecutionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  def _config(self, **overrides) -> xloop_lib.XLoopConfig:
    kwargs = dict(
        output_dir=self.output_dir,
        num_warmups=1,
        num_executions=2,
        run_optimized_code=False,
        in_process=True,
        use_chip_broker=False,
    )
    kwargs.update(overrides)
    return xloop_lib.XLoopConfig(**kwargs)

  def test_equivalent_implementations_match(self):
    [item] = xloop_lib.run_execution([_item("sum_sq")], self._config())
    self.assertEqual(
        item.execution_report.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
    )
    self.assertTrue(item.execution_report.custom_outputs_match_reference)
    self.assertGreater(item.execution_report.gmean_speedup_over_reference, 0.0)

  def test_a_wrong_implementation_is_caught(self):
    [item] = xloop_lib.run_execution(
        [_item("wrong", custom=_WRONG_ANSWER)], self._config()
    )
    # It ran fine; it is just incorrect. Those are different failures.
    self.assertEqual(
        item.execution_report.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
    )
    self.assertFalse(item.execution_report.custom_outputs_match_reference)

  def test_a_kernel_that_raises_is_reported_not_propagated(self):
    [item] = xloop_lib.run_execution(
        [_item("broken", custom=_RAISES_ON_IMPORT)], self._config()
    )
    self.assertEqual(
        item.execution_report.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_ERROR,
    )
    message = item.execution_results["default"].error_message
    self.assertIn("kernel is broken", message)

  def test_a_missing_entry_point_names_the_contract(self):
    [item] = xloop_lib.run_execution(
        [_item("no_entry", custom=_NO_ENTRY_POINT)], self._config()
    )
    message = item.execution_results["default"].error_message
    self.assertIn("get_model_identifier", message)

  def test_every_config_is_executed(self):
    item = _item(
        "multi",
        configs={"small": _INPUTS, "also_small": _INPUTS},
    )
    [executed] = xloop_lib.run_execution([item], self._config())
    self.assertCountEqual(
        executed.execution_results.keys(), ["small", "also_small"]
    )

  def test_first_config_only_runs_one(self):
    item = _item("multi", configs={"b": _INPUTS, "a": _INPUTS})
    [executed] = xloop_lib.run_execution(
        [item], self._config(first_config_only=True)
    )
    # Sorted, so the choice is reproducible rather than dict-order dependent.
    self.assertEqual(list(executed.execution_results.keys()), ["a"])

  def test_auxiliary_modules_are_importable(self):
    item = _item("aux", configs={"default": _INPUTS_VIA_AUX})
    config = item.input_generation_configs["default"]
    config.auxiliary_modules.modules["xloop_test_aux.shapes"] = _AUX_SHAPES
    [executed] = xloop_lib.run_execution([item], self._config())
    self.assertEqual(
        executed.execution_report.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
    )

  def test_an_item_without_configs_is_skipped_not_fatal(self):
    empty = xloop_item_pb2.XLoopItem(name="empty", id="empty")
    empty.reference_code = _SUM_OF_SQUARES
    executed = xloop_lib.run_execution([empty, _item("good")], self._config())
    self.assertEmpty(executed[0].execution_results)
    # The runnable item still ran.
    self.assertNotEmpty(executed[1].execution_results)

  def test_inputs_do_not_change_between_roles(self):
    # Both roles must see identical inputs or the comparison is meaningless.
    # A wrong answer on identical inputs is detectable; on different inputs
    # even a correct kernel would appear to fail.
    [item] = xloop_lib.run_execution([_item("sum_sq")], self._config())
    self.assertTrue(item.execution_report.custom_outputs_match_reference)

  def test_results_are_written_to_disk(self):
    xloop_lib.run_execution([_item("sum_sq")], self._config())
    results_path = os.path.join(
        self.output_dir, "sum_sq", "default", "results.json"
    )
    self.assertTrue(os.path.exists(results_path))
    with open(results_path) as f:
      results = json.load(f)
    self.assertEqual(results["config_name"], "default")
    self.assertEqual(results["num_executions"], 2)

  def test_input_items_are_not_mutated(self):
    original = _item("sum_sq")
    snapshot = xloop_item_pb2.XLoopItem()
    snapshot.CopyFrom(original)
    xloop_lib.run_execution([original], self._config())
    self.assertEqual(original, snapshot)


def _worker_binary() -> str | None:
  """Locates the worker binary in runfiles, if this environment has one."""
  srcdir = os.environ.get("TEST_SRCDIR")
  if not srcdir:
    return None
  relative = (
      "third_party/xprof/plugin/xprof/xloop/xloop_worker"
  )
  for prefix in ("google3", ""):
    candidate = os.path.join(srcdir, prefix, relative)
    if os.path.exists(candidate):
      return candidate
  return None


class SubprocessAndBrokerTest(absltest.TestCase):
  """Exercises the production path: real worker processes holding leases.

  The in-process tests bypass process spawning, device pinning and lease
  arbitration, which is most of what the orchestrator does.
  """

  def setUp(self):
    super().setUp()
    worker = _worker_binary()
    if worker is None:
      self.skipTest("worker binary not present in runfiles")
    self.output_dir = self.create_tempdir().full_path
    # No accelerators on a test machine; the broker takes a synthetic count so
    # the lease arbitration still runs for real.
    self.enter_context(
        absltest.mock.patch.dict(
            os.environ,
            {"XLOOP_NUM_CHIPS": "2", "XLOOP_WORKER_CMD": worker},
        )
    )

  def _config(self, **overrides) -> xloop_lib.XLoopConfig:
    kwargs = dict(
        output_dir=self.output_dir,
        num_warmups=1,
        num_executions=2,
        run_optimized_code=False,
        in_process=False,
        use_chip_broker=True,
    )
    kwargs.update(overrides)
    return xloop_lib.XLoopConfig(**kwargs)

  def test_a_worker_subprocess_produces_results(self):
    [item] = xloop_lib.run_execution([_item("sum_sq")], self._config())
    self.assertEqual(
        item.execution_report.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
        item.execution_results["default"].error_message,
    )
    self.assertTrue(item.execution_report.custom_outputs_match_reference)

  def test_more_tasks_than_chips_still_all_complete(self):
    items = [_item(f"item_{i}") for i in range(4)]
    executed = xloop_lib.run_execution(items, self._config())
    for item in executed:
      self.assertEqual(
          item.execution_report.execution_status,
          xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
          item.execution_results["default"].error_message,
      )

  def test_a_failing_worker_reports_its_stderr(self):
    [item] = xloop_lib.run_execution(
        [_item("broken", custom=_RAISES_ON_IMPORT)], self._config()
    )
    self.assertEqual(
        item.execution_report.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_ERROR,
    )
    # The message must name the kernel's own failure, not just an exit code.
    self.assertIn(
        "kernel is broken", item.execution_results["default"].error_message
    )

  def test_leases_are_released_after_the_run(self):
    xloop_lib.run_execution([_item("sum_sq")], self._config())
    payload = xloop_cli.xloop_chips()
    self.assertEmpty(payload["leased_chip_indices"])


class CliTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path
    self.item_path = os.path.join(self.output_dir, "in.json")
    input_utils.save_output_items([_item("sum_sq")], self.item_path)

  def _run(self, **overrides):
    kwargs = dict(
        item_path=self.item_path,
        output_dir=os.path.join(self.output_dir, "out"),
        num_warmups=1,
        num_executions=2,
        roles="reference,custom",
        in_process=True,
        use_chip_broker=False,
    )
    kwargs.update(overrides)
    return xloop_cli.xloop_run(**kwargs)

  def test_run_returns_a_summary_and_writes_items(self):
    payload = self._run()
    self.assertIn("sum_sq", payload["summary"])
    self.assertTrue(os.path.exists(payload["output_items_path"]))
    [record] = payload["items"]
    self.assertEqual(record["status"], "CODE_STATUS_EXECUTE_SUCCESS")
    self.assertTrue(record["custom_outputs_match_reference"])

  def test_an_unknown_role_is_rejected_before_any_work(self):
    with self.assertRaises(ValueError):
      self._run(roles="reference,typo")

  def test_detailed_summary_adds_detail(self):
    self.assertNotIn("detail", self._run())
    self.assertIn("detail", self._run(detailed_summary=True))

  def test_per_sample_times_are_measured_individually(self):
    # Regression: the harness used to report only an aggregate `active_sec`,
    # and the reader divided it evenly across samples. Every sample therefore
    # carried an identical time, the summarizer's interquartile spread was
    # exactly zero, and its faster/slower threshold silently collapsed to the
    # 2% floor on every run ever executed -- so any ratio outside 2% became a
    # confident verdict, including ones produced by transient interference.
    items = input_utils.load_input_items(self._run()["output_items_path"])
    result = items[0].execution_results["default"]
    self.assertGreater(len(result.sample_execution_results), 1)
    times = [
        s.reference_execution_time_sec for s in result.sample_execution_results
    ]
    self.assertTrue(all(t > 0 for t in times), times)
    self.assertGreater(
        len(set(times)), 1, f"per-sample times are all identical: {times}"
    )

  def test_the_noise_threshold_responds_to_real_spread(self):
    # Pins the gate as live end to end: a genuinely noisy arm must widen the
    # threshold above its floor. Asserting on compute_speedup_epsilon directly
    # would pass even with the aggregate-only plumbing that caused the bug.
    steady = [1.0] * 8
    noisy = [1.0, 1.9, 0.6, 1.7, 0.7, 1.8, 0.65, 1.85]
    self.assertEqual(
        summarize_results_lib.compute_speedup_epsilon(steady, steady),
        summarize_results_lib.MIN_SPEEDUP_EPSILON,
    )
    self.assertGreater(
        summarize_results_lib.compute_speedup_epsilon(steady, noisy),
        summarize_results_lib.MIN_SPEEDUP_EPSILON,
    )

  def test_run_reports_trace_dirs_that_exist(self):
    # Regression: this was keyed off a proto field nothing ever writes, so it
    # came back empty on real hardware even though the traces were on disk.
    # An empty mapping breaks the whole hand-off to the xprof analyses.
    [record] = self._run()["items"]
    trace_dirs = record["trace_dirs"]
    self.assertNotEmpty(trace_dirs)
    for key, path in trace_dirs.items():
      self.assertIn("/", key)
      self.assertTrue(os.path.isdir(path), f"{key} -> {path}")

  def test_summarize_reports_the_same_trace_dirs_as_run(self):
    payload = self._run()
    reread = xloop_cli.xloop_summarize(payload["output_items_path"])
    self.assertEqual(
        reread["items"][0]["trace_dirs"], payload["items"][0]["trace_dirs"]
    )

  def test_summarize_reproduces_the_run_summary(self):
    payload = self._run()
    reread = xloop_cli.xloop_summarize(payload["output_items_path"])
    self.assertEqual(reread["summary"], payload["summary"])

  def test_chips_reports_a_count_without_initializing_jax(self):
    payload = xloop_cli.xloop_chips()
    self.assertIn("num_chips", payload)
    self.assertIn("lease_dir", payload)
    self.assertGreaterEqual(payload["available"], 0)


if __name__ == "__main__":
  absltest.main()
