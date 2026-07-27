"""Tool to fetch kernel performance statistics and step times across 1P and 3P."""

from typing import Any, Literal

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import kernel_stats_tools


def compute_kernel_stats(
    source: Any,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown"] = "json",
    return_dict: bool = False,
    include_summary: bool = False,
    device_to_use: str | None = "TPU:0",
    trace_matchers: tuple[str, ...] | None = None,
) -> Any:
  """Unconditional real-time evaluation without caching or rate limiting.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, in-memory ProfileData/XSpace objects, or
  pre-computed statistical records.

  Args:
      source: XProf session ID, local file/directory path, serialized XSpace
        bytes, in-memory ProfileData/XSpace object, or pre-computed records.
      kernel_name: Optional specific tf_op_name / kernel name to filter by.
      limit: Number of top kernels to return when kernel_name is not provided.
      output_format: Output formatting specification ('json' or 'markdown').
      return_dict: If True, returns raw list of dict records (or enriched dict
        when include_summary is also True) instead of string.
      include_summary: If True, computes ground-truth timing via Disjoint
        Interval Union alongside per-kernel records.
      device_to_use: Device plane to target (e.g., "TPU:0").
      trace_matchers: Optional tuple of event name matchers for filtering.

  Returns:
      A formatted string representation or dictionary containing kernel
      statistics.
  """
  return kernel_stats_tools.get_kernel_stats(
      source,
      kernel_name=kernel_name,
      limit=limit,
      output_format=output_format,
      return_dict=return_dict,
      include_summary=include_summary,
      device_to_use=device_to_use,
      trace_matchers=trace_matchers,
  )


@decorators.cached(expire=86400)
@decorators.rate_limited(rate=1.0, burst=3)
def get_kernel_stats(
    source: Any,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown"] = "json",
    return_dict: bool = False,
    include_summary: bool = False,
    device_to_use: str | None = "TPU:0",
    trace_matchers: tuple[str, ...] | None = None,
) -> Any:
  """Fetches performance metrics for operations from XProf or local traces.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, in-memory ProfileData/XSpace objects, or
  pre-computed statistical records.

  Args:
      source: XProf session ID, local file/directory path, serialized XSpace
        bytes, in-memory ProfileData/XSpace object, or pre-computed records.
      kernel_name: Optional specific tf_op_name / kernel name to filter by.
      limit: Number of top kernels to return when kernel_name is not provided.
      output_format: Output formatting specification ('json' or 'markdown').
      return_dict: If True, returns raw list of dict records (or enriched dict
        when include_summary is also True) instead of string.
      include_summary: If True, computes ground-truth timing via Disjoint
        Interval Union alongside per-kernel records.
      device_to_use: Device plane to target (e.g., "TPU:0").
      trace_matchers: Optional tuple of event name matchers for filtering.

  Returns:
      A formatted string representation or dictionary containing kernel
      statistics.
  """
  return compute_kernel_stats(
      source,
      kernel_name=kernel_name,
      limit=limit,
      output_format=output_format,
      return_dict=return_dict,
      include_summary=include_summary,
      device_to_use=device_to_use,
      trace_matchers=trace_matchers,
  )


def compute_avg_step_time(
    source: Any,
    *,
    func_name: str | None = None,
    return_dict: bool = False,
) -> Any:
  """Unconditional step time calculation without caching or rate limiting.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, or in-memory ProfileData/XSpace objects.

  Args:
    source: Identifier or path for the profile data.
    func_name: Optional name of the function to filter stats for.
    return_dict: If True, returns a structured dictionary instead of text.

  Returns:
    Formatted summary string or performance statistics dictionary.
  """
  return kernel_stats_tools.get_avg_step_time(
      source,
      func_name=func_name,
      return_dict=return_dict,
  )


@decorators.cached(expire=86400)
@decorators.rate_limited(rate=1.0, burst=3)
def get_avg_step_time(
    source: Any,
    *,
    func_name: str | None = None,
    return_dict: bool = False,
) -> Any:
  """Computes average step and module time from XProf sessions or local trace logs.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, or in-memory ProfileData/XSpace objects.

  Args:
      source: XProf session ID, local file/directory path, serialized XSpace
        bytes, or in-memory ProfileData/XSpace object.
      func_name: Optional XLA module pattern or function substring to match.
      return_dict: If True, returns raw dict instead of JSON string.

  Returns:
      A JSON-formatted dictionary or raw dict containing avg_step_time_ms and
      step_count.
  """
  return compute_avg_step_time(
      source,
      func_name=func_name,
      return_dict=return_dict,
  )
