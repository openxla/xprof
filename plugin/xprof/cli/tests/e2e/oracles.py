"""Independent ground-truth oracle layer for XProf E2E trace analysis.

Provides three decoupled ground-truth oracles:
- Oracle 1 (O1): Raw XSpace plane walker (pure Python protobuf and timing
intervals).
- Oracle 2 (O2): HLO shape arithmetic and algorithmic roofline model.
- Oracle 3 (O3): Hardware datasheet physical invariant bounds checker.
"""

from collections.abc import Sequence
import dataclasses
import re
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  from xprof.cli.internal.oss import xplane_tools
except ImportError:
  from xprof.cli.internal.oss import xplane_tools


@dataclasses.dataclass(frozen=True)
class HardwareSpecs:
  """Hardware specification datasheet constants."""

  chip_type: str
  peak_tflops_bf16: float
  peak_tflops_f32: float
  peak_hbm_bw_gb_s: float
  hbm_capacity_gib: float
  ridge_point_flops_per_byte: float


HARDWARE_DATASHEET_TABLE: dict[str, HardwareSpecs] = {
    "TPU v6e": HardwareSpecs(
        chip_type="TPU v6e",
        peak_tflops_bf16=918.0,
        peak_tflops_f32=459.0,
        peak_hbm_bw_gb_s=1638.0,
        hbm_capacity_gib=32.0,
        ridge_point_flops_per_byte=918.0 * 1e12 / (1638.0 * 1e9),
    ),
    "TPU v5e": HardwareSpecs(
        chip_type="TPU v5e",
        peak_tflops_bf16=197.0,
        peak_tflops_f32=98.5,
        peak_hbm_bw_gb_s=819.0,
        hbm_capacity_gib=16.0,
        ridge_point_flops_per_byte=197.0 * 1e12 / (819.0 * 1e9),
    ),
    "TPU v4": HardwareSpecs(
        chip_type="TPU v4",
        peak_tflops_bf16=275.0,
        peak_tflops_f32=137.5,
        peak_hbm_bw_gb_s=1200.0,
        hbm_capacity_gib=32.0,
        ridge_point_flops_per_byte=275.0 * 1e12 / (1200.0 * 1e9),
    ),
}


class XSpaceOracle:
  """Oracle 1 (O1): Raw XSpace plane walker for ground-truth trace metrics."""

  def __init__(self, trace_source: Any):
    self._source = trace_source

  def compute_step_time_ms(self) -> float:
    """Computes average step time in ms from raw XSpace 'Steps' line."""
    # Check for TensorCore device planes (excluding SparseCore)
    # with exact "Steps" line
    for plane in xplane_tools.iter_planes(self._source):
      if (
          re.search(r"^/device:.*", plane.name)
          and "SparseCore" not in plane.name
      ):
        for line in plane.lines:
          if line.name.strip() == "Steps":
            durs = [
                e.duration_ns / 1_000_000.0
                for e in line.events
                if e.duration_ns > 10_000_000  # Filter out sub-step events
            ]
            if durs:
              return sum(durs) / len(durs)

    # Fallback to any plane with exact "Steps" line
    for plane in xplane_tools.iter_planes(self._source):
      for line in plane.lines:
        if line.name.strip() == "Steps":
          durs = [
              e.duration_ns / 1_000_000.0
              for e in line.events
              if e.duration_ns > 10_000_000
          ]
          if durs:
            return sum(durs) / len(durs)

    # Fallback to XLA Modules
    for plane in xplane_tools.iter_planes(self._source):
      for line in plane.lines:
        if "XLA MODULES" in line.name.upper():
          durs = [
              e.duration_ns / 1_000_000.0
              for e in line.events
              if e.duration_ns > 10_000_000
          ]
          if durs:
            return sum(durs) / len(durs)
    return 0.0

  def compute_device_active_duty_cycle(self) -> float:
    """Computes device active duty cycle using Disjoint Interval Union."""
    intervals: list[tuple[int, int]] = []
    ignored_lines = ("_counters_", "Power Throttle", "Steps", "XLA Modules")
    for plane in xplane_tools.iter_planes(self._source):
      if not re.search(r"^/device:.*", plane.name):
        continue
      for line in plane.lines:
        if any(ign.lower() in line.name.lower() for ign in ignored_lines):
          continue
        for event in line.events:
          start = int(event.start_ns)
          end = start + int(event.duration_ns)
          if end > start:
            intervals.append((start, end))

    if not intervals:
      return 0.0

    intervals.sort(key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    curr_start, curr_end = intervals[0]
    for s, e in intervals[1:]:
      if s <= curr_end:
        curr_end = max(curr_end, e)
      else:
        merged.append((curr_start, curr_end))
        curr_start, curr_end = s, e
    merged.append((curr_start, curr_end))

    total_active_ns = sum(e - s for s, e in merged)
    total_span_ns = merged[-1][1] - merged[0][0]
    if total_span_ns <= 0:
      return 0.0
    return min(1.0, total_active_ns / total_span_ns)

  def compute_collective_time_ms(self) -> float:
    """Computes total duration of collective communication operations."""
    collective_re = re.compile(
        r".*([Cc]ollective|[Aa]ll-reduce|[Aa]ll-to-all|[Rr]educe-scatter).*"
    )
    total_ns = 0
    for plane in xplane_tools.iter_planes(self._source):
      if not re.search(r"^/device:.*", plane.name):
        continue
      for line in plane.lines:
        for event in line.events:
          name = event.name
          if collective_re.search(name):
            total_ns += int(event.duration_ns)
    return total_ns / 1_000_000.0


class HloRooflineOracle:
  """Oracle 2 (O2): Algorithmic roofline and shape arithmetic oracle."""

  @classmethod
  def compute_matmul_flops(
      cls, batch_dims: Sequence[int], m: int, n: int, k: int
  ) -> float:
    """Computes theoretical algorithmic FLOPs for matrix multiplication."""
    batch_count = 1
    for d in batch_dims:
      batch_count *= d
    return float(2 * batch_count * m * n * k)

  @classmethod
  def compute_operational_intensity(
      cls, algorithmic_flops: float, logical_bytes: float
  ) -> float:
    """Calculates operational intensity (FLOP/byte)."""
    if logical_bytes <= 0:
      return 0.0
    return algorithmic_flops / logical_bytes

  @classmethod
  def compute_ridge_point(cls, peak_flops: float, peak_bw: float) -> float:
    """Computes operational ridge point (FLOP/byte) with 0.9313 TPU v6e derating."""
    return (0.9313 * peak_flops / peak_bw) if peak_bw > 0 else 0.0

  @classmethod
  def classify_roofline_bound(
      cls, operational_intensity: float, chip_type: str = "TPU v6e"
  ) -> str:
    """Classifies an operation as compute-bound or memory-bound."""
    specs = HARDWARE_DATASHEET_TABLE.get(
        chip_type, HARDWARE_DATASHEET_TABLE["TPU v6e"]
    )
    if operational_intensity >= specs.ridge_point_flops_per_byte:
      return "COMPUTE_BOUND"
    return "MEMORY_BOUND"


class DatasheetInvariantOracle:
  """Oracle 3 (O3): Hardware datasheet and physical invariant validator."""

  @classmethod
  def validate_metrics(
      cls,
      step_time_ms: float,
      duty_cycle: float,
      achieved_tflops: float | None = None,
      achieved_bw_gb_s: float | None = None,
      peak_memory_gib: float | None = None,
      chip_type: str = "TPU v6e",
  ) -> tuple[bool, list[str]]:
    """Validates metrics against physical invariants and datasheet limits."""
    specs = HARDWARE_DATASHEET_TABLE.get(
        chip_type, HARDWARE_DATASHEET_TABLE["TPU v6e"]
    )
    violations: list[str] = []

    if step_time_ms <= 0:
      violations.append(f"Step time must be positive (got {step_time_ms} ms)")
    if not (0.0 <= duty_cycle <= 1.0):
      violations.append(f"Duty cycle must be in [0, 1] (got {duty_cycle})")
    if (
        achieved_tflops is not None
        and achieved_tflops > specs.peak_tflops_bf16 * 1.05
    ):
      violations.append(
          f"Achieved TFLOPS ({achieved_tflops}) exceeds 105% peak"
          f" ({specs.peak_tflops_bf16})"
      )
    if (
        achieved_bw_gb_s is not None
        and achieved_bw_gb_s > specs.peak_hbm_bw_gb_s * 1.05
    ):
      violations.append(
          f"Achieved HBM BW ({achieved_bw_gb_s} GB/s) exceeds 105% peak"
          f" ({specs.peak_hbm_bw_gb_s} GB/s)"
      )
    if (
        peak_memory_gib is not None
        and peak_memory_gib > specs.hbm_capacity_gib * 1.05
    ):
      violations.append(
          f"Peak memory ({peak_memory_gib} GiB) exceeds capacity"
          f" ({specs.hbm_capacity_gib} GiB)"
      )

    return len(violations) == 0, violations
