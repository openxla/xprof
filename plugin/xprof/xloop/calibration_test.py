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
"""Calibration: proves the engine measures the kernel, not the dispatch path.

This test exists because the engine once shipped a timing harness that was
almost entirely wrong and passed every test in the suite. Wall-clock timing
around a jitted call includes Python, JAX dispatch, the driver and a
host-device round trip, which on a v6e cost ~160us against ~2.5us of actual
compute for a 1024-cube matmul. Every unit test still passed, because they all
checked plumbing -- that a number reached the right field -- and none checked
that the number meant anything.

The fixture runs a chain of K matmuls for K in {1, 8, 32, 128}, with the same
code on both sides of each comparison. Two properties follow, and together
they are hard to satisfy by accident:

  * The null property. Both sides are identical, so the reported speedup must
    be 1.0. This catches ordering bias: the reference arm runs first and pays
    the colder cache, which previously showed up as a 1.09x "speedup" between
    two arms doing exactly the same work.

  * The dynamic-range property. Device time for K=128 must be at least 10x
    that for K=1. This is the discriminator. A fixed overhead F added to real
    work t gives (F + 128t) / (F + t); for device time (F ~ 13us, t ~ 2.6us)
    that is ~26, while for wall clock (F ~ 160us) it is ~2.9. Note that an
    affine fit alone would *not* discriminate -- both are affine in K. Only
    the intercept relative to the slope separates them.

A second check, `TwoDeviceNullTest`, runs one item with identical arms that is
sharded over the 2 devices of one chip, with enough samples for the verdict to
have an interval. Its verdict must be "no change". This is the case where
running all samples of one arm before the other arm gave a "faster" verdict
between identical arms: one device drifted over the run, and the slowest
device sets each sample's time. Chips with one device each skip this check.

Requires a TPU, so it skips elsewhere rather than failing.
"""

import json
import os

from absl import logging
from absl.testing import absltest

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import chip_broker
from xprof.xloop import input_utils
from xprof.xloop import summarize_results_lib
from xprof.xloop import xloop_lib

_FIXTURE = "testdata/dispatch_floor_calibration.jsonl"

# Both arms run identical code, so anything outside this band is measurement
# bias rather than a real difference. Wide enough not to flake on a shared
# host, narrow enough to have caught the 1.09x ordering bias.
_NULL_TOLERANCE = 0.05

# See the module docstring: device time clears this comfortably, wall clock
# cannot reach it.
_MIN_DYNAMIC_RANGE = 10.0

_WARMUPS = 10
_SAMPLES = 20

# Enough samples for the paired interval on the median speedup to be narrow,
# so a drift between the arms would make the interval exclude 1.0.
_TWO_DEVICE_SAMPLES = 50

# Identical arms, sharded over every device the worker sees. On a chip with
# two devices each device computes half of the rows.
_SHARDED_MATMUL = """
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _matmul(x, w):
  mesh = Mesh(jax.devices(), ("d",))
  x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("d", None)))
  w = jax.lax.with_sharding_constraint(w, NamedSharding(mesh, P(None, None)))
  y = jnp.dot(x, w, preferred_element_type=jnp.float32)
  return jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("d", None)))


def get_model_identifier():
  return _matmul
"""

_SHARDED_MATMUL_INPUTS = """
import jax
import jax.numpy as jnp


def get_input_generator():
  def generate(key):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (8192, 4096), jnp.bfloat16)
    w = jax.random.normal(k2, (4096, 4096), jnp.bfloat16)
    return (x, w), {}
  return generate
"""


def _worker_binary() -> str | None:
  """Returns the worker binary beside this test, or None if it is absent.

  Under Bazel the worker is a data dependency of this test. There the test may
  have no usable `sys.executable` to start `python -m` with, so the worker must
  be named explicitly.

  Returns:
    The path to the worker binary, or None.
  """
  candidate = os.path.join(os.path.dirname(__file__), "xloop_worker")
  return candidate if os.path.exists(candidate) else None


def _has_tpu() -> bool:
  # Ask sysfs, not JAX: initializing JAX in this process would claim every
  # chip on the host, and each worker would then fail to open its own.
  return bool(chip_broker.discover_chips())


def _worker_env() -> dict[str, str]:
  """Returns the environment that points the engine at the worker binary."""
  worker = _worker_binary()
  return {} if worker is None else {"XLOOP_WORKER_CMD": worker}


class DispatchFloorCalibrationTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not _has_tpu():
      raise absltest.SkipTest("calibration requires a TPU")
    cls.enter_context(absltest.mock.patch.dict(os.environ, _worker_env()))

    items = input_utils.load_input_items(
        os.path.join(os.path.dirname(__file__), _FIXTURE)
    )

    config = xloop_lib.XLoopConfig(
        output_dir=absltest.TEST_TMPDIR.value,
        num_warmups=_WARMUPS,
        num_executions=_SAMPLES,
    )
    cls.executed = xloop_lib.run_execution(items, config)

  def _device_times(self, item) -> list[float]:
    """Reference per-sample times for an item's single config."""
    [result] = item.execution_results.values()
    return summarize_results_lib.sample_times(result, "reference")

  def test_every_item_ran(self):
    for item in self.executed:
      [result] = item.execution_results.values()
      self.assertEqual(
          result.execution_status,
          xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
          f"{item.name}: {result.error_message}",
      )

  def test_times_come_from_the_device_not_the_host(self):
    # A wall-clock fallback silently reintroduces the dispatch floor, so the
    # source matters as much as the value.
    for item in self.executed:
      [result] = item.execution_results.values()
      for role, source in result.timing_source.items():
        self.assertEqual(
            source, "trace", f"{item.name}/{role} fell back to wall clock"
        )

  def test_identical_implementations_report_no_speedup(self):
    for item in self.executed:
      speedup = item.execution_report.gmean_speedup_over_reference
      self.assertAlmostEqual(
          speedup,
          1.0,
          delta=_NULL_TOLERANCE,
          msg=(
              f"{item.name}: both arms run the same code, so {speedup:.3f}x is"
              " measurement bias, not a difference"
          ),
      )

  def test_device_time_has_the_dynamic_range_of_real_work(self):
    by_name = {item.name: item for item in self.executed}
    smallest = self._device_times(by_name["calibration_chain_1"])
    largest = self._device_times(by_name["calibration_chain_128"])

    # Medians: a single profiler stall would skew a mean badly.
    smallest.sort()
    largest.sort()
    small = smallest[len(smallest) // 2]
    large = largest[len(largest) // 2]
    self.assertGreater(small, 0.0)

    observed = large / small
    logging.info(
        "chain_1 %.2fus, chain_128 %.2fus, ratio %.1fx",
        small * 1e6,
        large * 1e6,
        observed,
    )
    self.assertGreater(
        observed,
        _MIN_DYNAMIC_RANGE,
        "128x the work produced only"
        f" {observed:.1f}x the measured time. A fixed per-call overhead is"
        " dominating the measurement, which is what this test exists to"
        " detect.",
    )

  def test_device_time_increases_with_work(self):
    ordered = []
    for k in (1, 8, 32, 128):
      times = self._device_times(
          {i.name: i for i in self.executed}[f"calibration_chain_{k}"]
      )
      times.sort()
      ordered.append((k, times[len(times) // 2]))
    for (k_small, t_small), (k_large, t_large) in zip(ordered, ordered[1:]):
      self.assertGreater(
          t_large,
          t_small,
          f"chain_{k_large} must take longer than chain_{k_small}",
      )


class TwoDeviceNullTest(absltest.TestCase):
  """Identical arms on the 2 devices of one chip must report "no change"."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not _has_tpu():
      raise absltest.SkipTest("calibration requires a TPU")
    cls.enter_context(absltest.mock.patch.dict(os.environ, _worker_env()))

    item = xloop_item_pb2.XLoopItem(
        name="two_device_null", id="two_device_null"
    )
    item.reference_code = _SHARDED_MATMUL
    item.custom_code = _SHARDED_MATMUL
    config = item.input_generation_configs["m8192_k4096_n4096"]
    config.input_generation_code = _SHARDED_MATMUL_INPUTS
    config.atol = 1e-2
    config.rtol = 1e-2

    cls.output_dir = os.path.join(absltest.TEST_TMPDIR.value, "two_device")
    # One chip per worker: the worker sees the devices of that chip only.
    [cls.executed] = xloop_lib.run_execution(
        [item],
        xloop_lib.XLoopConfig(
            output_dir=cls.output_dir,
            num_warmups=_WARMUPS,
            num_executions=_TWO_DEVICE_SAMPLES,
            run_optimized_code=False,
        ),
    )
    [cls.result] = cls.executed.execution_results.values()

  def setUp(self):
    super().setUp()
    self.assertEqual(
        self.result.execution_status,
        xloop_item_pb2.CODE_STATUS_EXECUTE_SUCCESS,
        self.result.error_message,
    )
    # The check needs the 2 devices of one chip. A chip with one device runs
    # the computation on one device, where the drift between devices that
    # this test guards against cannot happen.
    if self.result.environment.device_count < 2:
      self.skipTest(
          f"the leased chip has {self.result.environment.device_count} device;"
          " this check needs a chip with 2 devices"
      )

  def test_both_devices_are_timed_from_the_trace(self):
    for role in ("reference", "custom"):
      self.assertEqual(self.result.timing_source[role], "trace", role)
      self.assertNotIn(role, self.result.timing_notes)
    sample = self.result.sample_execution_results[0]
    self.assertLen(sample.reference_device_execution_times_sec, 2)

  def test_the_arms_alternate_in_blocks(self):
    with open(
        os.path.join(
            self.output_dir,
            "two_device_null",
            "m8192_k4096_n4096",
            "results.json",
        )
    ) as f:
      results = json.load(f)
    self.assertEqual(results["reference_block_sample_counts"], [25, 25])

  def test_identical_arms_report_no_change(self):
    report = self.executed.execution_report
    logging.info(
        "two-device null: speedup %.3f, interval %.3f-%.3f, band %.3f,"
        " verdict %r",
        report.gmean_speedup_over_reference,
        report.speedup_interval_low,
        report.speedup_interval_high,
        report.speedup_noise_band,
        report.verdict,
    )
    self.assertEqual(
        report.verdict,
        "no change",
        f"both arms run the same code, but the verdict is {report.verdict!r} at"
        f" {report.gmean_speedup_over_reference:.3f}x (interval"
        f" {report.speedup_interval_low:.3f}-{report.speedup_interval_high:.3f})."
        " A drift of one device over the run is being charged to one arm.",
    )


if __name__ == "__main__":
  absltest.main()
