"""Tool to fetch kernel performance statistics and step times across 1P and 3P."""

import json
from typing import Any, Literal
import warnings

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
    include_intra_kernel_regions: bool = False,
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
      output_format: Output format - 'json' (JSON string) or 'markdown'
        (markdown table string). 'dict' is deprecated and will be removed: it
        returns the parsed 'json' output.
      include_summary: If True, computes ground-truth timing via Disjoint
        Interval Union alongside per-kernel records.
      device_to_use: Device plane to target (e.g., "TPU:0").
      trace_matchers: Optional tuple of event name matchers for filtering.
      include_intra_kernel_regions: If True, also emits events from TPU lines
        holding regions *inside* a kernel ("LLO Ops", "Pallas Primitives",
        "<unit> Instructions"). Excluded by default because they overlap the
        kernel that contains them.
      bypass_cache: Whether to bypass cache.

  Returns:
      A JSON or markdown string containing kernel statistics, or the parsed
      JSON for the deprecated 'dict' format.

  Raises:
      ValueError: If neither source nor session_id is provided.
  """
  source = source or session_id
  if source is None:
    raise ValueError("Must provide either 'source' or 'session_id'.")
  if isinstance(source, (int, float)):
    source = str(source)
  legacy_dict = output_format == "dict"
  if legacy_dict:
    warnings.warn(
        "output_format='dict' is deprecated; use 'json' and json.loads().",
        DeprecationWarning,
        stacklevel=2,
    )
  engine_format: Literal["json", "markdown"] = (
      "markdown" if output_format == "markdown" else "json"
  )
  result = kernel_stats_tools.get_kernel_stats(
      source,
      kernel_name=kernel_name,
      limit=limit,
      output_format=engine_format,
      include_summary=include_summary,
      device_to_use=device_to_use,
      trace_matchers=trace_matchers,
      include_intra_kernel_regions=include_intra_kernel_regions,
      bypass_cache=bypass_cache,
  )
  return json.loads(result) if legacy_dict else result


@decorators.cached(expire=86400)
def get_kernel_stats(
    source: Any = None,
    session_id: str | None = None,
    *,
    kernel_name: str | None = None,
    limit: int = 10,
    output_format: Literal["json", "markdown"] = "json",
    include_summary: bool = False,
    device_to_use: str | None = "TPU:0",
    trace_matchers: tuple[str, ...] | None = None,
    include_intra_kernel_regions: bool = False,
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
      output_format: Output format - 'json' (JSON string) or 'markdown'
        (markdown table string).
      include_summary: If True, computes ground-truth timing via Disjoint
        Interval Union alongside per-kernel records.
      device_to_use: Device plane to target (e.g., "TPU:0").
      trace_matchers: Optional tuple of event name matchers for filtering.
      include_intra_kernel_regions: If True, also emits events from TPU lines
        holding regions *inside* a kernel ("LLO Ops", "Pallas Primitives",
        "<unit> Instructions"). Excluded by default because they overlap the
        kernel that contains them.
      bypass_cache: Whether to bypass cache.

  Returns:
      A JSON or markdown string containing kernel statistics.

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
      include_intra_kernel_regions=include_intra_kernel_regions,
      bypass_cache=bypass_cache,
  )
