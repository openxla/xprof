# Copyright 2025 The XProf Authors. All Rights Reserved.
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

Requires a TPU, so it skips elsewhere rather than failing.
"""

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


def _has_tpu() -> bool:
  # Ask sysfs, not JAX: initializing JAX in this process would claim every
  # chip on the host, and each worker would then fail to open its own.
  return bool(chip_broker.discover_chips())


class DispatchFloorCalibrationTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not _has_tpu():
      raise absltest.SkipTest("calibration requires a TPU")

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


if __name__ == "__main__":
  absltest.main()
