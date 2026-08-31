"""Tool to fetch kernel performance statistics and step times across 1P and 3P."""

from typing import Any, Literal

from xprof.cli.internal import decorators

from xprof.cli.internal.oss import kernel_stats_tools


def compute_kernel_stats(
    source: Any = None,
    session_id: str | None = None,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown", "dict"] = "json",
    include_summary: bool = False,
    device_to_use: str | None = "TPU:0",
    trace_matchers: tuple[str, ...] | None = None,
    bypass_cache: bool = False,
) -> Any:
  """Unconditional real-time evaluation without caching or rate limiting.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, in-memory ProfileData/XSpace objects, or
  pre-computed statistical records.

  Args:
      source: XProf session ID, local file/directory path, serialized XSpace
        bytes, in-memory ProfileData/XSpace object, or pre-computed records.
      session_id: Alias for source representing an XProf session ID or path.
      kernel_name: Optional specific tf_op_name / kernel name to filter by.
      limit: Number of top kernels to return when kernel_name is not provided.
      output_format: Output format - 'json' (JSON string), 'markdown' (markdown
        table string), or 'dict' (raw Python dict/list).
      include_summary: If True, computes ground-truth timing via Disjoint
        Interval Union alongside per-kernel records.
      device_to_use: Device plane to target (e.g., "TPU:0").
      trace_matchers: Optional tuple of event name matchers for filtering.
      bypass_cache: Whether to bypass cache.

  Returns:
      A formatted string representation or dictionary containing kernel
      statistics.

  Raises:
      ValueError: If neither source nor session_id is provided.
  """
  source = source or session_id
  if source is None:
    raise ValueError("Must provide either 'source' or 'session_id'.")
  if isinstance(source, (int, float)):
    source = str(source)
  return kernel_stats_tools.get_kernel_stats(
      source,
      kernel_name=kernel_name,
      limit=limit,
      output_format=output_format,
      include_summary=include_summary,
      device_to_use=device_to_use,
      trace_matchers=trace_matchers,
      bypass_cache=bypass_cache,
  )


@decorators.cached(expire=86400)
def get_kernel_stats(
    source: Any = None,
    session_id: str | None = None,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown", "dict"] = "json",
    include_summary: bool = False,
    device_to_use: str | None = "TPU:0",
    trace_matchers: tuple[str, ...] | None = None,
    bypass_cache: bool = False,
) -> Any:
  """Fetches performance metrics for operations from XProf or local traces.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, in-memory ProfileData/XSpace objects, or
  pre-computed statistical records.

  Args:
      source: XProf session ID, local file/directory path, serialized XSpace
        bytes, in-memory ProfileData/XSpace object, or pre-computed records.
      session_id: Alias for source representing an XProf session ID or path.
      kernel_name: Optional specific tf_op_name / kernel name to filter by.
      limit: Number of top kernels to return when kernel_name is not provided.
      output_format: Output format - 'json' (JSON string), 'markdown' (markdown
        table string), or 'dict' (raw Python dict/list).
      include_summary: If True, computes ground-truth timing via Disjoint
        Interval Union alongside per-kernel records.
      device_to_use: Device plane to target (e.g., "TPU:0").
      trace_matchers: Optional tuple of event name matchers for filtering.
      bypass_cache: Whether to bypass cache.

  Returns:
      A formatted string representation or dictionary containing kernel
      statistics.

  Raises:
      ValueError: If neither source nor session_id is provided.
  """
  source = source or session_id
  if source is None:
    raise ValueError("Must provide either 'source' or 'session_id'.")
  if isinstance(source, (int, float)):
    source = str(source)
  return compute_kernel_stats(
      source,
      kernel_name=kernel_name,
      limit=limit,
      output_format=output_format,
      include_summary=include_summary,
      device_to_use=device_to_use,
      trace_matchers=trace_matchers,
      bypass_cache=bypass_cache,
  )


def compute_avg_step_time(
    source: Any,
    *,
    func_name: str | None = None,
    output_format: Literal["json", "dict"] = "json",
    bypass_cache: bool = False,
) -> Any:
  """Unconditional step time calculation without caching or rate limiting.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, or in-memory ProfileData/XSpace objects.

  Args:
    source: Identifier or path for the profile data.
    func_name: Optional name of the function to filter stats for.
    output_format: Output format - 'json' (JSON string) or 'dict' (raw dict).
    bypass_cache: Whether to bypass cache.

  Returns:
    Formatted summary string or performance statistics dictionary.
  """
  if isinstance(source, (int, float)):
    source = str(source)
  return kernel_stats_tools.get_avg_step_time(
      source,
      func_name=func_name,
      output_format=output_format,
      bypass_cache=bypass_cache,
  )


@decorators.cached(expire=86400)
def get_avg_step_time(
    source: Any,
    *,
    func_name: str | None = None,
    output_format: Literal["json", "dict"] = "json",
    bypass_cache: bool = False,
) -> Any:
  """Computes average step and module time from XProf sessions or local trace logs.

  Supports polymorphic inputs: XProf session IDs (str), local file/directory
  paths, serialized XSpace bytes, or in-memory ProfileData/XSpace objects.

  Args:
      source: XProf session ID, local file/directory path, serialized XSpace
        bytes, or in-memory ProfileData/XSpace object.
      func_name: Optional XLA module pattern or function substring to match.
      output_format: Output format - 'json' (JSON string) or 'dict' (raw dict).
      bypass_cache: If True, skip cache lookup and storage.

  Returns:
      A JSON-formatted dictionary or raw dict containing avg_step_time_ms and
      step_count.
  """
  if isinstance(source, (int, float)):
    source = str(source)
  return compute_avg_step_time(
      source,
      func_name=func_name,
      output_format=output_format,
      bypass_cache=bypass_cache,
  )
