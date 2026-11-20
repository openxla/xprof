"""OSS hermetic backend to compute kernel stats from local XPlane traces."""

import collections
import functools
import json
import logging
import os
import re
import statistics as stats_mod
from typing import Any, Dict, List, Literal

from xprof.cli.internal import decorators
from xprof.cli.internal import trace_selection
from xprof.cli.internal.oss import xplane_tools
from xprof.cli.internal.oss import xprof_client

_DEVICE_PLANE_RE = re.compile(r"^/device:.*")

# Top-level accelerator chip planes, e.g. "/device:TPU:0" or "/device:GPU:3".
# Sub-planes of a chip (e.g. "/device:TPU:0 SparseCore 0") do not match.
_ACCELERATOR_PLANE_RE = re.compile(r"^/device:(TPU|GPU):(\d+)$", re.IGNORECASE)

# `device_to_use` value that aggregates every device plane on every host.
ALL_DEVICES = "all"

# Exact XLine names carrying *top-level* device kernels. Substring matching is
# unsafe here: with --xla_xprof_enable_custom_call_tracing=true the TPU tracer
# adds "Pallas Primitives" and "LLO Ops" lines holding the regions *inside* a
# kernel, and a substring test for "PALLAS" / "LLO OPS" matches those too,
# promoting every intra-kernel region to a top-level kernel. This mirrors the
# exact-name check already used by the kernel utilization converter.
_TPU_KERNEL_LINES = frozenset(("XLA OPS", "PALLAS"))

# Exact XLine names (and a suffix) carrying regions *within* a kernel. These
# overlap their containing kernel, so reporting them alongside it double counts
# and can rank a region above the kernel that contains it.
_TPU_INTRA_KERNEL_LINES = frozenset(("LLO OPS", "PALLAS PRIMITIVES"))
_TPU_INTRA_KERNEL_LINE_SUFFIX = " INSTRUCTIONS"

# Standard device line exclusion filters for non-computational metadata.
_DEVICE_LINE_EXCLUDE = (
    "COUNTER",
    "MODULES",
    "OVERLAY",
    "SYNC FLAG",
    "SENSOR",
)

# GPU kernel launch events carry their launch parameters in this stat, e.g.
# "regs:128 static_shared:0 dynamic_shared:12288 grid:2176,16,1 block:64,1,1
# occ_pct:25". Memcpy and memset events on the same streams do not, which is
# how the kernel stats converter tells kernels apart (GpuEventStats::IsKernel).
_KERNEL_DETAILS_STAT = "kernel_details"

# Stat names that mark a GPU stream event as a memory transfer or fill, mapped
# to the kind. The profiler writes "memcpy_details" and "Memset_details"
# (GetStatTypeStr in xplane_schema.cc); "memset_details" is also accepted.
_GPU_NON_KERNEL_STATS = {
    "memcpy_details": "memcpy",
    "Memset_details": "memset",
    "memset_details": "memset",
}

# Lower-case names of the derived timeline lines (GetDerivedLineName in
# xplane_schema and derived_timeline). Processed sessions add them to GPU
# device planes next to the `Stream #N` lines. Their events summarize the
# stream events again, so they are not kernels, memcpy or memset. A line from
# another thread can have a " - from #N" suffix.
_GPU_DERIVED_LINE_NAMES = frozenset((
    "steps",
    "framework name scope",
    "framework ops",
    "xla ops",
    "xla modules",
    "source code",
    "launch stats",
    "async xla ops",
    "host offload ops",
    "command buffers",
    "_counters_",
))
_DERIVED_LINE_SOURCE_SUFFIX = " - from "


def is_gpu_derived_line(line_name: str) -> bool:
  """Returns whether a GPU device line is a derived timeline line."""
  base = line_name.split(_DERIVED_LINE_SOURCE_SUFFIX, 1)[0].strip().lower()
  return base in _GPU_DERIVED_LINE_NAMES


def gpu_non_kernel_kind(stats: dict[str, Any]) -> str | None:
  """Returns 'memcpy' or 'memset' for a GPU transfer or fill event, else None."""
  for stat_name, kind in _GPU_NON_KERNEL_STATS.items():
    if stats.get(stat_name):
      return kind
  return None


# Kernel name substrings that mark a kernel as running on TensorCores. Mirrors
# kTensorCoreKernelNamePatterns in utils/cuda_type_utils.cc so the flags match
# the Kernel Stats tool. The list predates cuBLASLt `nvjet`, Triton
# `gemm_fusion` and cuDNN fused-attention kernels, so it reports False for them.
_TENSOR_CORE_KERNEL_NAME_PATTERNS = (
    "16816",
    "c1688",
    "conv1x1",
    "conv2d_c1_k1",
    "dgrad_1x1_stride_2x2",
    "direct_group",
    "first_layer_wgrad_kernel",
    "h1688",
    "h884",
    "hmma",
    "i16832",
    "i8816",
    "s884",
    "s1688",
    "xmma_gemm",
    "xmma_implicit_gemm",
    "xmma_sparse_conv",
    "xmma_sparse_gemm",
    "xmma_warp_specialized_implicit_gemm",
)

# Framework op name suffixes and substrings that can use TensorCores. Mirrors
# IsOpTensorCoreEligible in utils/kernel_stats_utils.cc.
_TENSOR_CORE_ELIGIBLE_OP_SUFFIXES = (
    "Conv2D",
    "Conv2DBackpropFilter",
    "Conv2DBackpropInput",
    "Conv3D",
    "DepthwiseConv2dNative",
    "DepthwiseConv2dNativeBackpropFilter",
    "DepthwiseConv2dNativeBackpropInput",
    "/MatMul",
    "FusedMatMul",
    "/CudnnRNN",
    "XlaDot",
    "XlaDotV2",
)
_TENSOR_CORE_ELIGIBLE_OP_SUBSTRINGS = (
    "BatchMatMul",
    "CudnnRNNV",
    "CudnnRNNForward",
    "CudnnRNNBackprop",
    "dot_general",
    "conv_general_dilated",
)


def _safe_stats(obj: Any) -> list[tuple[str, Any]]:
  """Returns (name, value) stat pairs, or [] when the object has none."""
  try:
    return [tuple(stat) for stat in obj.stats if len(tuple(stat)) == 2]
  except (AttributeError, TypeError):
    return []


def _parse_dim3(value: str) -> tuple[int, int, int] | None:
  parts = value.split(",")
  if len(parts) != 3 or not all(p.isdigit() for p in parts):
    return None
  return (int(parts[0]), int(parts[1]), int(parts[2]))


def parse_kernel_details(kernel_details: str) -> dict[str, Any]:
  """Parses a GPU `kernel_details` stat into launch parameters.

  Mirrors ParseKernelLaunchParams in utils/kernel_stats_utils.cc: unknown or
  malformed tokens are skipped and dimensions default to 1.

  Args:
    kernel_details: The stat value, e.g. "regs:32 static_shared:0
      dynamic_shared:0 grid:1,1,1 block:128,1,1 occ_pct:50".

  Returns:
    A dict with registers_per_thread, static_shmem_bytes, dynamic_shmem_bytes,
    block_dim, grid_dim and occupancy_pct (0.0 when absent).
  """
  params: dict[str, Any] = {
      "registers_per_thread": 0,
      "static_shmem_bytes": 0,
      "dynamic_shmem_bytes": 0,
      "block_dim": (1, 1, 1),
      "grid_dim": (1, 1, 1),
      "occupancy_pct": 0.0,
  }
  int_keys = {
      "regs": "registers_per_thread",
      "static_shared": "static_shmem_bytes",
      "dynamic_shared": "dynamic_shmem_bytes",
  }
  for token in kernel_details.split():
    key_value = token.split(":")
    if len(key_value) != 2:
      continue
    key, value = key_value
    if key in int_keys and value.isdigit():
      params[int_keys[key]] = int(value)
    elif key in ("block", "grid"):
      dims = _parse_dim3(value)
      if dims is not None:
        params[f"{key}_dim"] = dims
    elif key == "occ_pct":
      try:
        params["occupancy_pct"] = float(value)
      except ValueError:
        pass
  return params


@functools.lru_cache(maxsize=4096)
def is_kernel_using_tensor_core(kernel_name: str) -> bool:
  """Returns whether an NVIDIA kernel name marks a TensorCore kernel."""
  return any(p in kernel_name for p in _TENSOR_CORE_KERNEL_NAME_PATTERNS)


@functools.lru_cache(maxsize=4096)
def is_op_tensor_core_eligible(op_name: str) -> bool:
  """Returns whether a framework op can use TensorCores."""
  return op_name.endswith(_TENSOR_CORE_ELIGIBLE_OP_SUFFIXES) or any(
      s in op_name for s in _TENSOR_CORE_ELIGIBLE_OP_SUBSTRINGS
  )


@functools.lru_cache(maxsize=1024)
def _is_einsum_tensor_core_eligible(equation: str) -> bool:
  # Cached because each launch of a kernel repeats the same HLO equation.
  inputs, sep, _ = equation.partition("->")
  return bool(sep) and len(inputs.split(",")) == 2


class _GpuKernelAggregate:
  """Accumulates launch parameters of one GPU kernel across launches."""

  def __init__(self):
    # Launch parameter tuple -> [total duration ns, launches].
    self.launch_configs: dict[tuple[Any, ...], list[int]] = (
        collections.defaultdict(lambda: [0, 0])
    )
    # HLO op name -> total duration ns of the launches it made.
    self.hlo_op_ns: dict[str, int] = collections.defaultdict(int)
    self.uses_tensor_core = False
    self.tensor_core_eligible: bool | None = None

  def add(
      self,
      kernel_name: str,
      duration_ns: int,
      stats: dict[str, Any],
      is_nvidia: bool,
  ) -> None:
    """Adds one launch with its event stats."""
    params = parse_kernel_details(str(stats[_KERNEL_DETAILS_STAT]))
    occupancy_pct = params["occupancy_pct"]
    if not occupancy_pct:
      try:
        occupancy_pct = float(stats.get("theoretical_occupancy_pct") or 0.0)
      except ValueError:
        occupancy_pct = 0.0
    config = (
        params["registers_per_thread"],
        params["static_shmem_bytes"],
        params["dynamic_shmem_bytes"],
        params["block_dim"],
        params["grid_dim"],
        occupancy_pct,
    )
    entry = self.launch_configs[config]
    entry[0] += duration_ns
    entry[1] += 1
    if stats.get("hlo_op"):
      self.hlo_op_ns[str(stats["hlo_op"])] += duration_ns
    # Launches grouped under one name can come from different kernels, so a
    # single TensorCore launch marks the whole group.
    self.uses_tensor_core = self.uses_tensor_core or (
        is_nvidia and is_kernel_using_tensor_core(kernel_name)
    )
    tf_op = str(stats.get("tf_op") or "")
    if tf_op:
      op_name = tf_op.split(":", 1)[0]
      self.tensor_core_eligible = bool(
          self.tensor_core_eligible
          or is_op_tensor_core_eligible(op_name)
          or _is_einsum_tensor_core_eligible(str(stats.get("equation") or ""))
          or self.uses_tensor_core
      )

  @property
  def hlo_op(self) -> str:
    """Returns the HLO op with the most launch time, or "" when unknown.

    Several HLO ops can launch the same library kernel (e.g. one cuBLAS GEMM
    for two `dot` ops), and their launches share one record when no
    `tf_op_name` tells them apart.
    """
    if not self.hlo_op_ns:
      return ""
    return max(self.hlo_op_ns.items(), key=lambda item: (item[1], item[0]))[0]

  def to_fields(self) -> dict[str, Any]:
    """Returns record fields for the launch config with the most time."""
    config, (config_ns, _) = max(
        self.launch_configs.items(), key=lambda item: (item[1][0], item[0])
    )
    regs, static_shmem, dynamic_shmem, block, grid, occupancy_pct = config
    total_ns = sum(ns for ns, _ in self.launch_configs.values())
    fields = {
        "registers_per_thread": regs,
        "static_shmem_bytes": static_shmem,
        "dynamic_shmem_bytes": dynamic_shmem,
        "shmem_bytes": static_shmem + dynamic_shmem,
        "block_dim": list(block),
        "grid_dim": list(grid),
        # The tracer writes 0 when it did not compute occupancy (e.g. on
        # newer GPUs), so 0 means unknown rather than idle.
        "occupancy_pct": occupancy_pct or None,
        "num_launch_configs": len(self.launch_configs),
        "launch_config_share_pct": (
            round(100.0 * config_ns / total_ns, 2) if total_ns else 100.0
        ),
        "is_kernel_using_tensor_core": self.uses_tensor_core,
        "is_op_tensor_core_eligible": self.tensor_core_eligible,
    }
    if len(self.hlo_op_ns) > 1:
      fields["num_hlo_ops"] = len(self.hlo_op_ns)
    return fields


def classify_tpu_line(line_name: str) -> str:
  """Classifies a TPU device XLine by the granularity of events it carries.

  Args:
    line_name: The XLine name, in any case.

  Returns:
    'kernel' for top-level kernel lines, 'intra_kernel' for lines holding
    regions inside a kernel, and 'other' for everything else.
  """
  upper = line_name.upper()
  if upper in _TPU_KERNEL_LINES:
    return "kernel"
  if upper in _TPU_INTRA_KERNEL_LINES or upper.endswith(
      _TPU_INTRA_KERNEL_LINE_SUFFIX
  ):
    return "intra_kernel"
  return "other"


def resolve_device_planes(
    plane_names: list[str], device_to_use: str | None
) -> tuple[str, list[str]]:
  """Resolves `device_to_use` to the device plane names to aggregate.

  Args:
    plane_names: Names of every plane in the profile, in order. A name may
      repeat when the source holds several hosts and `device_to_use` is "all";
      the returned names are then de-duplicated, and callers sum every plane
      with a selected name.
    device_to_use: None picks the first accelerator chip plane, TPU before GPU
      and then by device index. "all" selects every device plane. Any other
      value names one chip, e.g. "GPU:0" or "/device:TPU:1" (case-insensitive
      device type); its sub-planes such as "/device:TPU:0 SparseCore 0" are
      included too.

  Returns:
    A (device, plane_names) pair: the resolved device plane name (or "all"),
    and the de-duplicated device plane names to aggregate.

  Raises:
    ValueError: If `device_to_use` names a device that is not in the profile.
  """
  device_names = list(
      dict.fromkeys(n for n in plane_names if _DEVICE_PLANE_RE.search(n))
  )
  if not device_names:
    return ALL_DEVICES, []
  if is_all_devices(device_to_use):
    return ALL_DEVICES, device_names
  if device_to_use is None:
    chips = []
    for name in device_names:
      match = _ACCELERATOR_PLANE_RE.match(name)
      if match:
        device_type_rank = 0 if match.group(1).upper() == "TPU" else 1
        chips.append((device_type_rank, int(match.group(2)), name))
    if not chips:
      return ALL_DEVICES, device_names
    chosen = min(chips)[2]
  else:
    target = device_to_use.strip()
    if not target.lower().startswith("/device:"):
      target = "/device:" + target
    chosen = next(
        (n for n in device_names if n.lower() == target.lower()), None
    )
    if chosen is None:
      raise ValueError(
          f"device_to_use={device_to_use!r} matches no device plane in the"
          f" profile. Available device planes: {device_names}. Pass"
          f" device_to_use={ALL_DEVICES!r} to aggregate every device plane."
      )
  selected = [
      n for n in device_names if n == chosen or n.startswith(chosen + " ")
  ]
  return chosen, selected


def is_all_devices(device_to_use: str | None) -> bool:
  """Returns True if `device_to_use` asks for every device on every host."""
  return (
      device_to_use is not None and device_to_use.strip().lower() == ALL_DEVICES
  )


def note_default_device(
    plane_names: list[str],
    device_to_use: str | None,
    device: str,
    *,
    host: str | None = None,
    num_hosts: int = 1,
    host_hint: str = "--host=<name> for another host",
) -> None:
  """Writes a stderr note when the result covers one of several chips or hosts.

  Args:
    plane_names: Names of the planes of the host that was read.
    device_to_use: The value passed by the caller.
    device: The device returned by `resolve_device_planes`.
    host: The label of the host that was read, if known.
    num_hosts: The number of hosts in the profile.
    host_hint: How to select another host, shown in the note.
  """
  if device == ALL_DEVICES:
    return
  num_chips = len({n for n in plane_names if _ACCELERATOR_PLANE_RE.match(n)})
  picked_chip = device_to_use is None and num_chips > 1
  if not picked_chip and num_hosts <= 1:
    return
  if num_hosts > 1:
    host_text = f" on host {host}" if host else " on the first host"
    scope = (
        f"{host_text} (host 1 of {num_hosts}; {num_chips} accelerator"
        " devices on that host)"
    )
    hint = (
        f"pass device_to_use='<TYPE>:<N>' for another device, {host_hint},"
        " or device_to_use='all' to sum every device on every host."
    )
  else:
    scope = f" of {num_chips} accelerator devices"
    hint = (
        "pass device_to_use='<TYPE>:<N>' for another device or 'all' to sum"
        " every device."
    )
  verb = "picked" if device_to_use is None else "read"
  decorators.emit_stderr_note(
      f"xprof-note: device_to_use={device_to_use} {verb} {device}{scope};"
      f" {hint}\n"
  )


def _trace_paths(source: Any) -> list[str] | None:
  """Returns the trace files a path or session source reads, else None.

  Mirrors the source handling of `xplane_tools.iter_planes`.

  Args:
    source: The `get_kernel_stats` source.

  Returns:
    The sorted trace paths for a file, directory, or session ID source, or
    None for bytes and in-memory sources.

  Raises:
    FileNotFoundError: If a path source does not exist or holds no traces.
  """
  if hasattr(source, "planes") or isinstance(source, bytes):
    return None
  if isinstance(source, os.PathLike):
    source = str(source)
  if not isinstance(source, str):
    return None
  if (
      os.path.exists(source)
      or source.startswith("/")
      or source.endswith((".xplane.pb", ".xspace.pb"))
  ):
    if not os.path.exists(source):
      raise FileNotFoundError(f"Path does not exist: {source!r}")
    if os.path.isfile(source):
      return [source]
  paths = list(xprof_client.get_client().get_xspace_paths(source))
  if not paths:
    raise FileNotFoundError(f"No .xplane.pb or .xspace.pb files in {source!r}")
  return paths


def load_host_planes(
    source: Any, all_hosts: bool
) -> tuple[list[Any], str | None, int]:
  """Loads the planes of the first host, or of every host if `all_hosts`.

  A directory or session with one trace file per host reads only the first
  file (in sorted order) unless `all_hosts` is set. A single XSpace that holds
  several hosts (the same device plane name more than once) keeps the first
  plane of each name unless `all_hosts` is set.

  Args:
    source: The `get_kernel_stats` source.
    all_hosts: Whether to load every host.

  Returns:
    A (planes, host, num_hosts) tuple: the loaded planes, the label of the
    host read ("all" when every host of several was read, None when unknown),
    and the number of hosts in the source.
  """
  paths = _trace_paths(source)
  if paths is not None:
    num_hosts = len(paths)
    if num_hosts > 1 and not all_hosts:
      paths_to_read = paths[:1]
      host = trace_selection.trace_labels(paths)[0]
    else:
      paths_to_read = paths
      host = (
          ALL_DEVICES if num_hosts > 1 else trace_selection.trace_stem(paths[0])
      )
    planes = []
    for path in paths_to_read:
      planes.extend(xplane_tools.iter_planes(path))
    return planes, host, num_hosts

  planes = list(xplane_tools.iter_planes(source))
  chip_counts = collections.Counter(
      p.name for p in planes if _ACCELERATOR_PLANE_RE.match(p.name)
  )
  num_hosts = max(chip_counts.values(), default=1)
  if num_hosts <= 1:
    return planes, None, 1
  if all_hosts:
    return planes, ALL_DEVICES, num_hosts
  seen: set[str] = set()
  first_host_planes = []
  for plane in planes:
    if plane.name not in seen:
      seen.add(plane.name)
      first_host_planes.append(plane)
  return first_host_planes, None, num_hosts


def device_label(device: str, host: str | None, num_hosts: int) -> str:
  """Returns `device` with the host it was read from on multi-host sources."""
  if num_hosts <= 1 or device == ALL_DEVICES:
    return device
  if host and host != ALL_DEVICES:
    return f"{device} on host {host} (1 of {num_hosts})"
  return f"{device} on the first host (1 of {num_hosts})"


def compute_disjoint_interval_union_ns(intervals: list[tuple[int, int]]) -> int:
  """Computes total duration from overlapping intervals using sweep-line merge.

  This prevents double-counting concurrent operations across MXU compute and
  DMA/ICI transfer planes, which inflates active hardware compute time by
  up to 47.3% on modern accelerators (b/537646336).

  Args:
    intervals: A list of (start_ns, end_ns) tuples.

  Returns:
    The total non-overlapping duration in nanoseconds.
  """
  if not intervals:
    return 0
  sorted_intervals = sorted(intervals, key=lambda x: x[0])
  merged = []
  curr_start, curr_end = sorted_intervals[0]
  for start, end in sorted_intervals[1:]:
    if start <= curr_end:
      curr_end = max(curr_end, end)
    else:
      merged.append((curr_start, curr_end))
      curr_start, curr_end = start, end
  merged.append((curr_start, curr_end))
  return sum(max(0, end - start) for start, end in merged)


def _interval_totals(intervals: list[tuple[int, int]]) -> dict[str, Any]:
  """Returns the count, summed duration and union duration of intervals."""
  return {
      "count": len(intervals),
      "total_duration_us": round(
          sum(max(0, end - start) for start, end in intervals) / 1000.0, 4
      ),
      "wall_duration_us": round(
          compute_disjoint_interval_union_ns(intervals) / 1000.0, 4
      ),
  }


def format_markdown_table(
    df_records: List[Dict[str, Any]],
    kernel_name: str | None,
    device: str | None = None,
) -> str:
  """Formats kernel stats as a markdown table.

  Args:
    df_records: Kernel records to render.
    kernel_name: The kernel the records were filtered to, if any.
    device: The device the records were aggregated over, if known; shown under
      the heading.

  Returns:
    The markdown table.
  """
  lines = []
  if kernel_name:
    lines.append(f"# Kernel Stats for `{kernel_name}`")
  else:
    lines.append("# Top Kernels by Duration")
  lines.append("")
  if device:
    lines.append(f"Device: `{device}`")
    lines.append("")
  has_gpu = any("registers_per_thread" in row for row in df_records)
  header = (
      "| Kernel | Total Duration (us) | Execution Count | Avg Duration (us) |"
  )
  align = "| :--- | ---: | ---: | ---: |"
  if has_gpu:
    header += (
        " Occupancy (%) | Regs/Thread | Shared Mem (B) | Block | Grid |"
        " TensorCore Used | TC Eligible |"
    )
    align += " ---: | ---: | ---: | :--- | :--- | :--- | :--- |"
  lines.append(header)
  lines.append(align)
  for row in df_records:
    total_dur = float(row.get("total_duration_us", 0.0) or 0.0)
    avg_dur = float(row.get("avg_duration_us", 0.0) or 0.0)
    line = (
        f"| `{row.get('kernel_name', 'unknown')}` | {total_dur:.2f} |"
        f" {row.get('execution_count', 0)} | {avg_dur:.2f} |"
    )
    if has_gpu:
      line += _format_gpu_columns(row)
    lines.append(line)
  return "\n".join(lines)


def _format_gpu_columns(row: Dict[str, Any]) -> str:
  """Formats the GPU launch columns of one markdown row."""
  if "registers_per_thread" not in row:
    return " | | | | | | |"

  def dims(value: Any) -> str:
    return "x".join(str(v) for v in value) if value else ""

  def flag(value: Any) -> str:
    return "" if value is None else ("yes" if value else "no")

  occupancy = row.get("occupancy_pct")
  configs = row.get("num_launch_configs", 1)
  block = dims(row.get("block_dim"))
  if configs and configs > 1:
    block += f" (+{configs - 1} configs)"
  return (
      f" {'n/a' if occupancy is None else f'{occupancy:g}'} |"
      f" {row.get('registers_per_thread', '')} |"
      f" {row.get('shmem_bytes', '')} | {block} |"
      f" {dims(row.get('grid_dim'))} |"
      f" {flag(row.get('is_kernel_using_tensor_core'))} |"
      f" {flag(row.get('is_op_tensor_core_eligible'))} |"
  )


def _apply_kernel_truncation(
    records: list[dict[str, Any]],
    *,
    kernel_name: str | None,
    limit: int,
    total_duration_us_all: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Computes truncation fields and applies `limit` to kernel records."""
  total_records = len(records)
  non_region = [
      r for r in records if not r.get("is_intra_kernel_region", False)
  ]
  sliced = records[:limit] if not kernel_name and limit > 0 else list(records)
  truncation_fields = decorators.build_truncation_fields(
      total_records=total_records,
      returned_records=len(sliced),
      top_level_records=len(non_region),
      total_field="total_duration_us_all_records",
      total_value=total_duration_us_all,
  )
  return truncation_fields, sliced


def _format_kernel_records_output(
    records: list[dict[str, Any]],
    *,
    truncation_fields: dict[str, Any],
    kernel_name: str | None,
    limit: int,
    output_format: Literal["json", "markdown"],
    device: str | None = None,
) -> str:
  """Formats kernel records as markdown or JSON and emits truncation notes."""
  truncated = bool(truncation_fields["truncated"])
  total_records = int(truncation_fields["total_records"])
  raw_total_dur = truncation_fields.get("total_duration_us_all_records")
  total_dur_all = float(raw_total_dur) if raw_total_dur is not None else None
  if output_format == "markdown":
    table = format_markdown_table(records, kernel_name, device)
    if truncated:
      if total_dur_all is not None:
        table += (
            f"\n\nShowing {len(records)} of {total_records} kernels"
            f" (--limit={limit}). Total duration across all outermost"
            f" kernels: {total_dur_all:.2f} us (sums across device planes,"
            " excluding nested loop body ops and intra-kernel regions; pass"
            " --include_summary=True for wall-clock"
            " total_device_duration_us). Do not sum the rows above to get a"
            " total."
        )
      else:
        table += (
            f"\n\nShowing {len(records)} of {total_records} kernels"
            f" (--limit={limit}). Total duration across all kernels is"
            " unavailable for pre-computed records without event timestamps."
            " Do not sum the rows above to get a total."
        )
    return table
  if truncated:
    # The JSON result is a plain list, so report truncation on stderr, in
    # the same style as the `xprof-capture:` line, instead of changing its
    # shape. `include_summary=True` returns these fields in the object.
    if total_dur_all is not None:
      decorators.emit_stderr_note(
          f"xprof-note: truncated: returned {len(records)} of"
          f" {total_records} kernels (--limit={limit});"
          f" total_duration_us_all_records={total_dur_all}"
          " (sum of outermost kernel durations across device planes,"
          " excluding nested loop body ops and intra-kernel regions). Do"
          " not sum the rows to get a total; pass --include_summary=True"
          " for total_device_duration_us and truncation fields.\n"
      )
    else:
      decorators.emit_stderr_note(
          f"xprof-note: truncated: returned {len(records)} of"
          f" {total_records} kernels (--limit={limit});"
          " total_duration_us_all_records=null (unavailable for"
          " pre-computed records without event timestamps). Do not sum the"
          " rows to get a total; pass --include_summary=True for"
          " truncation fields.\n"
      )
  return json.dumps(records, indent=2)


def _matches(pattern: str, text: str) -> bool:
  """Matches text against a regex pattern or literal substring."""
  try:
    if re.search(pattern, text):
      return True
  except re.error:
    pass
  return pattern in text


def get_kernel_stats(
    source: Any,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown"] = "json",
    include_summary: bool = False,
    device_to_use: str | None = None,
    trace_matchers: tuple[str, ...] | None = None,
    include_intra_kernel_regions: bool = False,
    bypass_cache: bool = False,
) -> Any:
  """Computes performance metrics for operations from local XPlanes in OSS.

  Supports polymorphic inputs: session IDs, file paths, bytes, or in-memory
  ProfileData/XSpace objects. When include_summary=True, the JSON output is an
  enriched summary object containing ground-truth device durations via Disjoint
  Interval Union, step durations, and stats.

  Args:
    source: XProf session ID, local file/directory path, serialized XSpace
      bytes, in-memory ProfileData/XSpace object, or pre-computed records.
    kernel_name: Optional specific tf_op_name / kernel name to filter by.
    limit: Number of top kernels to return when kernel_name is not provided.
    output_format: Output format - 'json' (JSON string) or 'markdown' (markdown
      table string).
    include_summary: If True, computes and returns ground-truth timing via
      Disjoint Interval Union alongside per-kernel records.
    device_to_use: Device to aggregate. None (default) picks the first
      accelerator chip plane of the first host, TPU before GPU and then by index
      (e.g. "/device:GPU:0" on a GPU profile). Pass a device such as "TPU:1" or
      "GPU:3" to select that device on the first host, or "all" to sum every
      device plane on every host. The first host is the first trace file in
      sorted order for a directory or session, and the first copy of each device
      plane for a single XSpace that holds several hosts; use a single trace
      file (or the CLI `--host=<name>`) for another host. Raises ValueError if
      the named device is not in the profile. Ignored for pre-computed record
      inputs.
    trace_matchers: Optional tuple of event name matchers for filtering.
    include_intra_kernel_regions: If True, also emits events from TPU lines that
      carry regions *inside* a kernel ("LLO Ops", "Pallas Primitives", "<unit>
      Instructions"). These overlap the kernel that contains them, so they are
      excluded by default on trace inputs to keep the ranking over kernels only.
      Included records are tagged with is_intra_kernel_region=True. Pre-computed
      record inputs keep all supplied rows unchanged.
    bypass_cache: Whether to bypass cache.

  Returns:
      A JSON string (a list of kernel records, or a summary object when
      include_summary=True) or a markdown table string. The summary object
      also has `truncated`, `total_records`, `top_level_records` (count of
      kernel records excluding intra-kernel regions), `returned_records`, and
      `total_duration_us_all_records` (sum of outermost kernel event durations
      across device planes before `limit`, excluding ops nested inside
      control-flow kernels such as `%while` and excluding intra-kernel regions,
      or `None` when called with pre-computed records that lack event
      timestamps; see `total_device_duration_us` for wall-clock device time).
      When `limit` drops rows, the list form writes an `xprof-note: truncated`
      line to stderr and the markdown form appends a "Showing N of M kernels"
      line. The summary object also has `device`, `device_planes`, `host`
      (the label of the host read, "all" when every host of several was read,
      or None when unknown) and `num_hosts`.
  """
  del bypass_cache
  if isinstance(source, (int, float)):
    source = str(source)
  try:
    if isinstance(source, (dict, list)):
      raw_records = (
          source.get("kernel_records", source)
          if isinstance(source, dict)
          else source
      )
      if not isinstance(raw_records, list):
        raw_records = [raw_records] if isinstance(raw_records, dict) else []
      # Pre-computed records carry no event start times, so nested loop body
      # ops cannot be distinguished from outermost kernels; leave
      # total_duration_us_all_records as None rather than summing records.
      records = [
          r
          for r in raw_records
          if isinstance(r, dict)
          and (not kernel_name or r.get("kernel_name") == kernel_name)
      ]
      truncation_fields, records = _apply_kernel_truncation(
          records,
          kernel_name=kernel_name,
          limit=limit,
          total_duration_us_all=None,
      )
      if include_summary:
        summary_out = dict(source) if isinstance(source, dict) else {}
        summary_out["kernel_records"] = records
        summary_out.update(truncation_fields)
        return json.dumps(summary_out, indent=2)
      return _format_kernel_records_output(
          records,
          truncation_fields=truncation_fields,
          kernel_name=kernel_name,
          limit=limit,
          output_format=output_format,
      )

    kernel_durations_us = collections.defaultdict(list)
    top_level_total_us = 0.0
    all_intervals: list[tuple[int, int]] = []
    line_intervals: dict[str, list[tuple[int, int]]] = collections.defaultdict(
        list
    )
    custom_call_intervals: list[tuple[int, int]] = []
    step_durations_us: list[float] = []
    excluded_region_lines: set[str] = set()
    gpu_kernels: dict[str, _GpuKernelAggregate] = collections.defaultdict(
        _GpuKernelAggregate
    )
    # Memcpy or memset kind -> (start_ns, end_ns) of each event.
    non_kernel_intervals: dict[str, list[tuple[int, int]]] = (
        collections.defaultdict(list)
    )
    excluded_derived_lines: set[str] = set()
    # The non-kernel totals cover the whole device, so they are not comparable
    # with a total that a name filter limited to some kernels.
    filters_active = bool(kernel_name or trace_matchers)

    planes, host, num_hosts = load_host_planes(
        source, all_hosts=is_all_devices(device_to_use)
    )
    plane_names = [plane.name for plane in planes]
    device, device_plane_names = resolve_device_planes(
        plane_names, device_to_use
    )
    selected_planes = frozenset(device_plane_names)
    if host is not None:
      host_hint = "--host=<name> for another host"
    else:
      host_hint = (
          "a single-host source for another host (this XSpace holds every"
          " host, and only the first copy of each device plane is read)"
      )
    note_default_device(
        plane_names,
        device_to_use,
        device,
        host=host,
        num_hosts=num_hosts,
        host_hint=host_hint,
    )

    for plane in planes:
      if plane.name not in selected_planes:
        continue

      is_tpu = "TPU" in plane.name.upper()
      is_gpu = "GPU" in plane.name.upper()
      is_nvidia = is_gpu and any(
          name == "device_vendor" and str(value).lower() == "nvidia"
          for name, value in _safe_stats(plane)
      )

      for line in plane.lines:
        line_name_upper = line.name.upper()
        if include_summary:
          for event in line.events:
            start_ns = int(event.start_ns)
            end_ns = start_ns + int(event.duration_ns)
            line_intervals[line.name].append((start_ns, end_ns))

        is_region_line = False
        if is_tpu:
          line_kind = classify_tpu_line(line.name)
          if line_kind == "intra_kernel":
            # Regions inside a kernel overlap their container, so they are not
            # top-level kernels and must not be ranked against one.
            if not include_intra_kernel_regions:
              excluded_region_lines.add(line.name)
              continue
            is_region_line = True
          elif line_kind != "kernel":
            # Still check XLA Modules for step durations if include_summary.
            if include_summary and "XLA MODULES" in line_name_upper:
              for event in line.events:
                step_durations_us.append(float(event.duration_ns) / 1000.0)
            continue
        elif any(w in line_name_upper for w in _DEVICE_LINE_EXCLUDE):
          continue
        elif is_gpu and is_gpu_derived_line(line.name):
          # Derived lines repeat the stream events at op, module or step
          # granularity, so their events are neither kernels nor transfers.
          excluded_derived_lines.add(line.name)
          continue

        # Per (plane, line): an event is outermost unless an earlier event on
        # the same line still covers its start (e.g. a `%while` op covering the
        # ops in its body on `XLA Ops`).
        cover_end_ns = -1
        for event in sorted(
            line.events,
            key=lambda e: (int(e.start_ns), -int(e.duration_ns)),
        ):
          start_ns = int(event.start_ns)
          end_ns = start_ns + int(event.duration_ns)
          # Materialize the stats once: they are scanned several times below.
          event_stats = tuple(event.stats)
          gpu_stats = None
          if is_gpu:
            gpu_stats = dict(event_stats)
            if not gpu_stats.get(_KERNEL_DETAILS_STAT):
              # Memcpy and memset activity share streams with kernels but are
              # not kernels; the kernel stats converter skips them too. Other
              # events without launch parameters are skipped without a count.
              kind = gpu_non_kernel_kind(gpu_stats)
              if kind is not None and not filters_active:
                non_kernel_intervals[kind].append((start_ns, end_ns))
              continue
          name_info = event.name
          tf_op_name = None
          for stat_name, stat_val in event_stats:
            if stat_name == "tf_op_name" and stat_val:
              tf_op_name = str(stat_val)
              break
          if tf_op_name:
            name_info = tf_op_name
          elif name_info.isdigit():
            for stat_name, stat_val in event_stats:
              if stat_val and stat_name in (
                  "msg",
                  "message",
                  "annotation",
                  "label",
              ):
                name_info = str(stat_val)
                break

          if kernel_name and name_info != kernel_name:
            continue

          # Apply trace matchers if provided.
          if trace_matchers:
            if not any(_matches(m, name_info) for m in trace_matchers):
              continue

          dur_us = float(event.duration_ns) / 1000.0
          kernel_durations_us[(name_info, is_region_line)].append(dur_us)
          if gpu_stats is not None:
            gpu_kernels[name_info].add(
                event.name, int(event.duration_ns), gpu_stats, is_nvidia
            )
          if not is_region_line and start_ns >= cover_end_ns:
            top_level_total_us += dur_us
          cover_end_ns = max(cover_end_ns, end_ns)

          # Region intervals are nested inside kernel intervals, so including
          # them would not change the union but would blur its meaning.
          if include_summary and not is_region_line:
            all_intervals.append((start_ns, end_ns))
            name_lower = name_info.lower()
            if (
                name_lower.startswith("custom-call")
                or "custom_call" in name_lower
                or "pallas" in name_lower
                or "PALLAS" in line_name_upper
            ):
              custom_call_intervals.append((start_ns, end_ns))

    if not kernel_durations_us:
      msg = f"No kernel stats found for session {source}"
      if kernel_name:
        msg += f" and kernel {kernel_name}"
      if device != ALL_DEVICES:
        msg += f" on device {device_label(device, host, num_hosts)}"
      if output_format == "markdown":
        return f"# Info\n{msg}\n"
      return json.dumps({"info": msg}, indent=2)

    records = []
    for (name, is_region), dur_list in kernel_durations_us.items():
      count = len(dur_list)
      total_us = sum(dur_list)
      avg_us = total_us / count if count > 0 else 0.0
      # Extract compact canonical name (e.g. 'fusion.668')
      canonical = name.strip().split(":")[0].strip().split()[0] if name else ""
      record = {
          "kernel_name": name,
          "canonical_name": canonical,
          "short_name": canonical,
          "hlo_op_name": canonical,
          "raw_name": name,
          "total_duration_us": round(total_us, 4),
          "execution_count": count,
          "avg_duration_us": round(avg_us, 4),
          "is_intra_kernel_region": is_region,
      }
      gpu_kernel = gpu_kernels.get(name)
      if gpu_kernel is not None:
        if gpu_kernel.hlo_op:
          record["hlo_op_name"] = gpu_kernel.hlo_op
        record.update(gpu_kernel.to_fields())
      records.append(record)

    records.sort(key=lambda x: x["total_duration_us"], reverse=True)
    # `top_level_total_us` sums outermost kernel events per (plane, line)
    # before `limit` drops any rows, so ops inside `%while` bodies and
    # intra-kernel regions are not double-counted.
    truncation_fields, records = _apply_kernel_truncation(
        records,
        kernel_name=kernel_name,
        limit=limit,
        total_duration_us_all=top_level_total_us,
    )

    if include_summary:
      total_ns = compute_disjoint_interval_union_ns(all_intervals)
      total_us = float(total_ns / 1000.0)
      total_ms = float(total_ns / 1_000_000.0)
      by_line_duration_ns = {
          line_name: compute_disjoint_interval_union_ns(intervals)
          for line_name, intervals in line_intervals.items()
      }
      custom_call_ns = compute_disjoint_interval_union_ns(custom_call_intervals)
      custom_call_us = round(float(custom_call_ns / 1000.0), 4)
      custom_call_pct = (
          round((custom_call_ns / total_ns) * 100.0, 2) if total_ns > 0 else 0.0
      )
      mean_us = total_us if not step_durations_us else (
          sum(step_durations_us) / len(step_durations_us)
      )
      std_us = 0.0
      if len(step_durations_us) > 1:
        std_us = stats_mod.stdev(step_durations_us)
      summary: dict[str, Any] = {
          "total_device_duration_ns": total_ns,
          "total_device_duration_us": total_us,
          "total_device_duration_ms": total_ms,
          "by_line_duration_ns": by_line_duration_ns,
          "custom_call_duration_us": custom_call_us,
          "custom_call_share_pct": custom_call_pct,
          "kernel_records": records,
          "step_durations_us": step_durations_us,
          "stats": {"mean_us": round(mean_us, 4), "std_us": round(std_us, 4)},
          "device": device,
          "device_planes": device_plane_names,
          "host": host,
          "num_hosts": num_hosts,
          **truncation_fields,
      }
      if non_kernel_intervals:
        by_kind = {
            kind: _interval_totals(intervals)
            for kind, intervals in sorted(non_kernel_intervals.items())
        }
        every_interval = []
        for intervals in non_kernel_intervals.values():
          every_interval.extend(intervals)
        # Transfers can run long before or after the kernels (e.g. weight
        # loads), so only the part inside the kernel span is comparable with
        # total_device_duration_us.
        kernel_start_ns = min((s for s, _ in all_intervals), default=0)
        kernel_end_ns = max((e for _, e in all_intervals), default=0)
        # Clipping keeps start <= end because only intervals that overlap
        # [kernel_start_ns, kernel_end_ns] are kept.
        in_kernel_span = [
            (max(start, kernel_start_ns), min(end, kernel_end_ns))
            for start, end in every_interval
            if start < kernel_end_ns and end > kernel_start_ns
        ]
        summary["excluded_gpu_non_kernel_events"] = {
            **_interval_totals(every_interval),
            "kernel_span_us": round(
                max(0, kernel_end_ns - kernel_start_ns) / 1000.0, 4
            ),
            "wall_duration_in_kernel_span_us": round(
                compute_disjoint_interval_union_ns(in_kernel_span) / 1000.0, 4
            ),
            "by_kind": by_kind,
            "note": (
                "Memcpy and memset events on GPU streams carry no kernel"
                " launch parameters and are not kernels, so they are not"
                " ranked or counted in total_device_duration_us."
                " total_duration_us sums events on every stream, so"
                " overlapping streams count more than once. wall_duration_us"
                " is their union in time over the whole capture, which can"
                " be much longer than the kernel span. Compare"
                " wall_duration_in_kernel_span_us (the union between the"
                " first kernel start and the last kernel end) with"
                " total_device_duration_us."
            ),
        }
      elif filters_active and any(
          "GPU" in name.upper() for name in device_plane_names
      ):
        summary["excluded_gpu_non_kernel_events_note"] = (
            "excluded_gpu_non_kernel_events is omitted because kernel_name"
            " or trace_matchers limits total_device_duration_us to some"
            " kernels, while memcpy and memset totals cover the whole device."
            " Run without these filters to compare transfers with kernels."
        )
      if excluded_derived_lines:
        summary["excluded_gpu_derived_lines"] = sorted(excluded_derived_lines)
      if excluded_region_lines:
        summary["excluded_intra_kernel_region_lines"] = sorted(
            excluded_region_lines
        )
        summary["note"] = (
            "Events on intra-kernel region lines were excluded because they"
            " overlap the kernel that contains them and would otherwise be"
            " ranked as top-level kernels. Pass"
            " include_intra_kernel_regions=True to include them, tagged with"
            " is_intra_kernel_region."
        )
      return json.dumps(summary, indent=2)

    return _format_kernel_records_output(
        records,
        truncation_fields=truncation_fields,
        kernel_name=kernel_name,
        limit=limit,
        output_format=output_format,
        device=(
            None
            if is_all_devices(device)
            else device_label(device, host, num_hosts)
        ),
    )

  except (FileNotFoundError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error in OSS get_kernel_stats for source %r", source
    )
    raise RuntimeError(f"Failed to get kernel stats: {e}") from e
