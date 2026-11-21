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
"""End-to-end tests for xloop_lib and xloop_cli.

Runs real JAX kernels through the worker on CPU. These are the tests that
would catch a break in the contract between the item proto, the worker's
module loading, and the harness.
"""

import io
import json
import os
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import chip_broker
from xprof.xloop import input_utils
from xprof.xloop import summarize_results_lib
from xprof.xloop import xloop_cli
from xprof.xloop import xloop_lib
from xprof.xloop import xloop_worker_lib

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

# A bfloat16 kernel, for the ULP report. The reference squares in float32 and
# rounds once. The custom kernel takes another float32 path, so a product near
# a rounding boundary can round to the next bfloat16 value: at most 1 ULP.
_BF16_SQUARE = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    return (x * x).astype(jnp.bfloat16)

  return f
"""

_BF16_SQUARE_VIA_LOG = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    return jnp.exp(2.0 * jnp.log(jnp.abs(x))).astype(jnp.bfloat16)

  return f
"""

# The bug the ULP report must make obvious: the last tile is never written.
_BF16_SQUARE_DROPS_THE_TAIL = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    y = (x * x).astype(jnp.bfloat16)
    return y.at[16:].set(0)

  return f
"""

_BF16_SQUARE_WITH_NAN_AND_INF = """
import jax.numpy as jnp


def get_model_identifier():
  def f(x):
    y = (x * x).astype(jnp.bfloat16)
    return y.at[0].set(jnp.nan).at[1].set(jnp.inf)

  return f
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


def _read_results(
    output_dir: str, item_name: str, config_name: str = "default"
) -> dict[str, Any]:
  """Reads the results.json that the worker wrote for one task."""
  with open(
      os.path.join(output_dir, item_name, config_name, "results.json")
  ) as f:
    return json.load(f)


class ChipCountFromTopologyTest(absltest.TestCase):

  def test_empty_topology_is_unspecified(self):
    self.assertIsNone(xloop_lib._chip_count_from_topology(""))

  def test_shape_is_multiplied_out(self):
    self.assertEqual(xloop_lib._chip_count_from_topology("2x4"), 8)

  def test_bare_shape_is_accepted(self):
    self.assertEqual(xloop_lib._chip_count_from_topology("2x2x1"), 4)

  def test_garbage_is_unspecified(self):
    self.assertIsNone(xloop_lib._chip_count_from_topology("not-a-topology"))


class SafePathComponentTest(absltest.TestCase):

  def test_a_safe_name_is_unchanged(self):
    self.assertEqual(
        xloop_lib.safe_path_component("matmul-v2.bf16"), "matmul-v2.bf16"
    )

  def test_unsafe_names_become_one_component(self):
    for name in ("../../x", "/tmp/x", "a/b", "..", ".", "", "x" * 300):
      with self.subTest(name=name):
        safe = xloop_lib.safe_path_component(name)
        self.assertNotIn(os.sep, safe)
        self.assertNotIn(safe, ("", ".", ".."))
        self.assertLessEqual(len(safe), 128)
        # Applying it again changes nothing.
        self.assertEqual(xloop_lib.safe_path_component(safe), safe)

  def test_names_that_differ_only_in_unsafe_characters_do_not_collide(self):
    names = ["a/b", "a?b", "a_b", "a b"]
    safe = {xloop_lib.safe_path_component(n) for n in names}
    self.assertLen(safe, len(names))

  def test_task_dirs_stay_under_the_output_dir(self):
    root = self.create_tempdir().full_path
    real_root = os.path.realpath(root)
    for label in ("../../x", "/tmp/x", "a/b"):
      for config_name in ("../../y", "/etc", "c/d"):
        path = os.path.realpath(xloop_lib.task_dir(root, label, config_name))
        self.assertEqual(os.path.dirname(os.path.dirname(path)), real_root)

  def test_a_symlink_out_of_the_output_dir_is_refused(self):
    root = self.create_tempdir().full_path
    outside = self.create_tempdir().full_path
    os.symlink(outside, os.path.join(root, "item"))
    with self.assertRaisesRegex(ValueError, "outside"):
      xloop_lib.task_dir(root, "item", "default")


class WriteModuleTest(absltest.TestCase):

  def test_a_dotted_name_becomes_a_package_tree(self):
    work_dir = self.create_tempdir().full_path
    path = xloop_worker_lib._write_module(work_dir, "pkg.sub.mod", "X = 1\n")
    self.assertEqual(path, os.path.join(work_dir, "pkg", "sub", "mod.py"))
    self.assertTrue(
        os.path.exists(os.path.join(work_dir, "pkg", "sub", "__init__.py"))
    )

  def test_a_name_that_is_not_a_module_name_is_refused(self):
    work_dir = self.create_tempdir().full_path
    for name in ("../evil", "a/b", "a..b", "", "pkg.class", "/abs.mod"):
      with self.subTest(name=name):
        with self.assertRaisesRegex(ValueError, "not a dotted Python"):
          xloop_worker_lib._write_module(work_dir, name, "")
    # Nothing was written, inside or outside the work directory.
    self.assertEmpty(os.listdir(work_dir))

  def test_a_symlink_out_of_the_work_dir_is_refused(self):
    work_dir = self.create_tempdir().full_path
    outside = self.create_tempdir().full_path
    os.symlink(outside, os.path.join(work_dir, "pkg"))
    with self.assertRaisesRegex(ValueError, "outside"):
      xloop_worker_lib._write_module(work_dir, "pkg.mod", "")
    self.assertEmpty(os.listdir(outside))


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

  def test_unsafe_item_and_config_names_stay_under_the_output_dir(self):
    item = _item("../../escape", configs={"../up": _INPUTS, "a/b": _INPUTS})
    [executed] = xloop_lib.run_execution([item], self._config())
    self.assertLen(executed.execution_results, 2)
    for result in executed.execution_results.values():
      self.assertEqual(
          result.execution_status,
          xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
          result.error_message,
      )
    # Everything was written below the output directory.
    self.assertFalse(
        os.path.exists(os.path.join(os.path.dirname(self.output_dir), "up"))
    )
    [label_dir] = os.listdir(self.output_dir)
    self.assertLen(os.listdir(os.path.join(self.output_dir, label_dir)), 2)

  def test_an_unsafe_auxiliary_module_name_fails_the_task(self):
    item = _item("aux_bad", configs={"default": _INPUTS})
    config = item.input_generation_configs["default"]
    config.auxiliary_modules.modules["../evil"] = "X = 1\n"
    [executed] = xloop_lib.run_execution([item], self._config())
    result = executed.execution_results["default"]
    self.assertNotEqual(
        result.execution_status, xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS
    )
    self.assertIn("not a dotted Python", result.error_message)

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
    self.assertEqual(results["num_blocks"], 2)

  def test_num_blocks_reaches_the_harness(self):
    xloop_lib.run_execution(
        [_item("sum_sq")], self._config(num_executions=3, num_blocks=3)
    )
    with open(
        os.path.join(self.output_dir, "sum_sq", "default", "results.json")
    ) as f:
      results = json.load(f)
    self.assertEqual(results["num_blocks"], 3)
    for role in ("reference", "custom"):
      self.assertEqual(results[f"{role}_block_sample_counts"], [1, 1, 1])

  def test_input_items_are_not_mutated(self):
    original = _item("sum_sq")
    snapshot = xloop_item_pb2.XLoopItem()
    snapshot.CopyFrom(original)
    xloop_lib.run_execution([original], self._config())
    self.assertEqual(original, snapshot)

  def test_without_xparity_no_ulp_report_is_written(self):
    [item] = xloop_lib.run_execution([_item("sum_sq")], self._config())
    results = _read_results(self.output_dir, "sum_sq")
    self.assertFalse(results["xparity"])
    self.assertNotIn("custom_outputs_ulp", results)
    self.assertEmpty(item.execution_results["default"].outputs_ulp)
    self.assertEmpty(item.execution_report.outputs_max_ulp)

  def test_xparity_reaches_the_harness_in_process(self):
    [item] = xloop_lib.run_execution(
        [_item("sum_sq")], self._config(xparity=True)
    )
    results = _read_results(self.output_dir, "sum_sq")
    self.assertTrue(results["xparity"])
    self.assertEqual(results["custom_outputs_ulp"]["status"], "ok")
    # The reference is the baseline, so it gets no report of its own.
    self.assertNotIn("reference_outputs_ulp", results)
    self.assertIn("custom", item.execution_report.outputs_max_ulp)

  def test_a_correct_bf16_kernel_is_within_a_few_ulps(self):
    item = _item("bf16", reference=_BF16_SQUARE, custom=_BF16_SQUARE_VIA_LOG)
    # 1 bfloat16 ULP is up to 0.8% relative error, which the default
    # tolerance rejects. The item states the tolerance it needs.
    config = item.input_generation_configs["default"]
    config.atol = 1e-3
    config.rtol = 2e-2
    [executed] = xloop_lib.run_execution([item], self._config(xparity=True))
    report = executed.execution_results["default"].outputs_ulp["custom"]
    self.assertTrue(executed.execution_report.custom_outputs_match_reference)
    self.assertEqual(report.status, "ok")
    self.assertEqual(report.dtype, "bfloat16")
    self.assertLessEqual(report.max_ulp, 2)
    self.assertTrue(report.within_contract)
    self.assertEqual(report.nan_count, 0)

  def test_a_dropped_tail_tile_is_far_over_the_contract(self):
    item = _item(
        "tail", reference=_BF16_SQUARE, custom=_BF16_SQUARE_DROPS_THE_TAIL
    )
    [executed] = xloop_lib.run_execution([item], self._config(xparity=True))
    report = executed.execution_results["default"].outputs_ulp["custom"]
    # `correct` still comes from the tolerance check, not from the report.
    self.assertFalse(executed.execution_report.custom_outputs_match_reference)
    self.assertGreater(report.max_ulp, 1000)
    self.assertFalse(report.within_contract)
    self.assertGreaterEqual(report.worst_index[0], 16)
    self.assertEqual(report.worst_output_value, 0.0)

  def test_nan_and_inf_outputs_are_reported(self):
    item = _item(
        "nonfinite",
        reference=_BF16_SQUARE,
        custom=_BF16_SQUARE_WITH_NAN_AND_INF,
    )
    [executed] = xloop_lib.run_execution([item], self._config(xparity=True))
    report = executed.execution_results["default"].outputs_ulp["custom"]
    self.assertFalse(executed.execution_report.custom_outputs_match_reference)
    self.assertGreaterEqual(report.nan_count, 1)
    self.assertGreaterEqual(report.inf_count, 1)
    self.assertEqual(report.reference_nan_count, 0)
    self.assertGreaterEqual(report.non_finite_mismatch_count, 2)

  def test_xparity_does_not_change_the_verdict(self):
    off = xloop_lib.run_execution(
        [_item("wrong", custom=_WRONG_ANSWER)],
        self._config(output_dir=os.path.join(self.output_dir, "off")),
    )
    on = xloop_lib.run_execution(
        [_item("wrong", custom=_WRONG_ANSWER)],
        self._config(
            output_dir=os.path.join(self.output_dir, "on"), xparity=True
        ),
    )
    self.assertEqual(
        off[0].execution_report.custom_outputs_match_reference,
        on[0].execution_report.custom_outputs_match_reference,
    )
    self.assertEqual(
        off[0].execution_report.execution_status,
        on[0].execution_report.execution_status,
    )


def _worker_binary() -> str | None:
  """Locates the worker binary in runfiles, if this environment has one.

  The worker is a data dependency of this test, so it sits beside this file in
  the runfiles tree whatever the workspace is called.

  Returns:
    The path to the worker binary, or None if it is not present.
  """
  candidate = os.path.join(os.path.dirname(__file__), "xloop_worker")
  return candidate if os.path.exists(candidate) else None


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
      result = item.execution_results["default"]
      self.assertEqual(
          item.execution_report.execution_status,
          xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
          result.error_message,
      )
      # Each result names the chip it ran on, one of the two on the host.
      self.assertIn(result.leased_chip, ("synthetic:0", "synthetic:1"))
      # Synthetic chips have no device node.
      self.assertEqual(result.leased_device_path, "")

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

  def test_num_blocks_reaches_the_worker_process(self):
    output_dir = os.path.join(self.output_dir, "blocks")
    xloop_lib.run_execution(
        [_item("sum_sq")],
        self._config(output_dir=output_dir, num_executions=3, num_blocks=3),
    )
    with open(
        os.path.join(output_dir, "sum_sq", "default", "results.json")
    ) as f:
      results = json.load(f)
    self.assertEqual(results["num_blocks"], 3)
    self.assertEqual(results["custom_block_sample_counts"], [1, 1, 1])

  def test_leases_are_released_after_the_run(self):
    xloop_lib.run_execution([_item("sum_sq")], self._config())
    payload = xloop_cli.xloop_chips()
    self.assertEmpty(payload["leased_chip_indices"])

  def test_xparity_reaches_the_worker_process(self):
    output_dir = os.path.join(self.output_dir, "xparity")
    [item] = xloop_lib.run_execution(
        [_item("sum_sq")], self._config(output_dir=output_dir, xparity=True)
    )
    results = _read_results(output_dir, "sum_sq")
    self.assertTrue(results["xparity"])
    self.assertEqual(results["custom_outputs_ulp"]["status"], "ok")
    self.assertIn("custom", item.execution_report.outputs_max_ulp)

  def test_a_busy_host_reports_busy_not_an_error(self):
    # Another process holds every chip on the host.
    holder = chip_broker.ChipBroker()
    with holder.lease(timeout_sec=1.0), holder.lease(timeout_sec=1.0):
      outcome = xloop_lib.run_execution_with_outcome(
          [_item("sum_sq")],
          self._config(max_local_workers=1, lease_timeout_sec=0.0),
      )
    [item] = outcome.items
    result = item.execution_results["default"]
    self.assertEqual(result.execution_status, xloop_item_pb2.CODE_STATUS_BUSY)
    self.assertEqual(result.leased_chip, "")
    [busy] = outcome.busy
    self.assertEqual(busy["config"], "default")
    self.assertEqual(busy["capacity"], 2)
    self.assertGreaterEqual(busy["queue_position"], 1)

  def test_a_capped_run_uses_a_chip_another_run_left_free(self):
    # Regression: a cap of one worker used to mean "chip 0 only", so several
    # capped runs on one host all queued on chip 0.
    peer = chip_broker.ChipBroker()
    with peer.lease(timeout_sec=1.0) as held:
      self.assertEqual(held.resource_id, "synthetic:0")
      for overrides in ({"max_local_workers": 1}, {"tpu_topology": "1x1"}):
        with self.subTest(**overrides):
          [item] = xloop_lib.run_execution(
              [_item("sum_sq")],
              self._config(lease_timeout_sec=0.0, **overrides),
          )
          result = item.execution_results["default"]
          self.assertEqual(
              result.execution_status,
              xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
              result.error_message,
          )
          self.assertEqual(result.leased_chip, "synthetic:1")


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
    self.assertEqual(payload["status"], "OK")
    self.assertEmpty(payload["busy"])
    self.assertIn("sum_sq", payload["summary"])
    self.assertNotIn("max ULP", payload["summary"])
    self.assertTrue(os.path.exists(payload["output_items_path"]))
    [record] = payload["items"]
    self.assertEqual(record["status"], "CODE_STATUS_EXECUTE_SUCCESS")
    self.assertTrue(record["custom_outputs_match_reference"])
    self.assertEmpty(record["outputs_max_ulp"])

  def test_run_passes_xparity(self):
    payload = self._run(xparity=True)
    self.assertIn("max ULP", payload["summary"])
    [record] = payload["items"]
    self.assertIn("custom", record["outputs_max_ulp"])
    reread = xloop_cli.xloop_summarize(payload["output_items_path"])
    self.assertEqual(reread["summary"], payload["summary"])

  def test_run_reports_a_busy_host_as_structured_json(self):
    worker = _worker_binary()
    if worker is None:
      self.skipTest("worker binary not present in runfiles")
    self.enter_context(
        mock.patch.dict(
            os.environ, {"XLOOP_NUM_CHIPS": "1", "XLOOP_WORKER_CMD": worker}
        )
    )
    holder = chip_broker.ChipBroker(num_chips=1)
    with holder.lease(timeout_sec=1.0):
      payload = self._run(
          in_process=False, use_chip_broker=True, lease_timeout_sec=0.0
      )
    self.assertEqual(payload["status"], "BUSY")
    [busy] = payload["busy"]
    self.assertEqual(busy["config"], "default")
    self.assertGreaterEqual(busy["queue_position"], 1)
    self.assertEqual(busy["timeout_sec"], 0.0)
    [record] = payload["items"]
    self.assertEqual(record["status"], "CODE_STATUS_BUSY")
    # The payload is what `xloop run` prints, so it must be valid JSON.
    self.assertEqual(json.loads(json.dumps(payload)), payload)

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

  def test_run_passes_num_blocks(self):
    self._run(num_executions=3, num_blocks=3)
    with open(
        os.path.join(
            self.output_dir, "out", "sum_sq", "default", "results.json"
        )
    ) as f:
      results = json.load(f)
    self.assertEqual(results["reference_block_sample_counts"], [1, 1, 1])

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

  def test_main_prints_json(self):
    stdout = io.StringIO()
    with mock.patch.object(sys, "stdout", stdout):
      xloop_cli.main(["xloop", "chips"])
    payload = json.loads(stdout.getvalue())
    self.assertEqual(payload, xloop_cli.xloop_chips())
    self.assertIn("num_chips", payload)

  def test_main_keeps_subcommand_flags(self):
    stdout = io.StringIO()
    with mock.patch.object(sys, "stdout", stdout):
      xloop_cli.main([
          "xloop",
          "summarize",
          self._run()["output_items_path"],
          "--detailed_summary",
      ])
    payload = json.loads(stdout.getvalue())
    self.assertIn("detail", payload)
    self.assertEqual(payload["items"][0]["name"], "sum_sq")

  def _roles_through_fire(self, roles_flag: str) -> tuple[bool, bool, bool]:
    """Runs `xloop run` through Fire and returns the roles the engine got."""
    with mock.patch.object(
        xloop_lib,
        "run_execution_with_outcome",
        autospec=True,
        return_value=xloop_lib.RunOutcome(items=[], busy=[]),
    ) as run, mock.patch.object(sys, "stdout", io.StringIO()):
      xloop_cli.main([
          "xloop",
          "run",
          self.item_path,
          os.path.join(self.output_dir, "out"),
          roles_flag,
      ])
    config = run.call_args.args[1]
    return (
        config.run_reference_code,
        config.run_custom_code,
        config.run_optimized_code,
    )

  def test_main_accepts_an_unquoted_comma_list_of_roles(self):
    # Regression: Fire parses an unquoted `a,b` as a tuple, and the documented
    # `--roles=reference,custom` crashed on `tuple.split`.
    self.assertEqual(
        self._roles_through_fire("--roles=reference,custom"),
        (True, True, False),
    )

  def test_main_accepts_a_quoted_comma_list_of_roles(self):
    self.assertEqual(
        self._roles_through_fire("--roles='reference,custom'"),
        (True, True, False),
    )
    self.assertEqual(
        self._roles_through_fire('--roles="reference,optimized"'),
        (True, False, True),
    )

  def test_main_accepts_a_single_role(self):
    self.assertEqual(
        self._roles_through_fire("--roles=custom"), (False, True, False)
    )

  def test_run_accepts_roles_as_a_sequence(self):
    run = self.enter_context(
        mock.patch.object(
            xloop_lib,
            "run_execution_with_outcome",
            autospec=True,
            return_value=xloop_lib.RunOutcome(items=[], busy=[]),
        )
    )
    self._run(roles=("reference", "custom"))
    self._run(roles=["custom", " "])
    first, second = [c.args[1] for c in run.call_args_list]
    self.assertTrue(first.run_reference_code and first.run_custom_code)
    self.assertFalse(first.run_optimized_code)
    self.assertFalse(second.run_reference_code or second.run_optimized_code)
    self.assertTrue(second.run_custom_code)

  def test_chips_reports_a_count_without_initializing_jax(self):
    payload = xloop_cli.xloop_chips()
    self.assertIn("num_chips", payload)
    self.assertIn("lease_dir", payload)
    self.assertGreaterEqual(payload["available"], 0)


if __name__ == "__main__":
  absltest.main()
