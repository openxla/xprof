"""OSS tool to diff two XProf sessions or local trace paths."""

from collections.abc import Callable
import json
import logging
from typing import Any

from xprof.cli.tools import get_hlo_stats_tool
from xprof.cli.tools import get_kernel_stats_tool
from xprof.cli.tools import get_kpi_metrics_tool


def _fetch_optional_json(
    tool_name: str, session_id: str, fetch: Callable[[], str]
) -> dict[str, Any]:
  """Runs an optional stats fetch, returning {} and logging on failure.

  Args:
    tool_name: Name of the tool being called, for the log message.
    session_id: Session ID or path the tool is called on, for the log message.
    fetch: Zero-argument callable returning the tool's JSON string output.

  Returns:
    The parsed JSON object, or {} if the fetch failed or did not return a JSON
    object.
  """
  try:
    result = json.loads(fetch())
  except Exception:  # pylint: disable=broad-exception-caught
    # Any tool or backend error only drops this optional section of the diff.
    logging.warning(
        "diff_sessions: %s failed for %s; omitting it from the diff.",
        tool_name,
        session_id,
        exc_info=True,
    )
    return {}
  return result if isinstance(result, dict) else {}


def diff_sessions(
    *,
    baseline_session_id: str,
    optimized_session_id: str,
    bypass_cache: bool = False,
) -> str:
  """Diffs two XProf sessions or local trace directories using kernel and device stats.

  Args:
    baseline_session_id: Baseline session ID or local trace path.
    optimized_session_id: Optimized session ID or local trace path.
    bypass_cache: Whether to bypass cache.

  Returns:
    JSON-formatted comparison summary with device time delta and per-kernel
    diffs.
  """
  try:
    base_summary = json.loads(
        get_kernel_stats_tool.get_kernel_stats(
            baseline_session_id,
            include_summary=True,
            bypass_cache=bypass_cache,
        )
    )
    opt_summary = json.loads(
        get_kernel_stats_tool.get_kernel_stats(
            optimized_session_id,
            include_summary=True,
            bypass_cache=bypass_cache,
        )
    )
    if not isinstance(base_summary, dict):
      base_summary = {}
    if not isinstance(opt_summary, dict):
      opt_summary = {}

    # KPI and HLO stats are optional enrichments: each fetch is guarded
    # separately so a failure on one session does not suppress the other's
    # metrics or abort the kernel diff, and the failure is logged.
    base_kpi = _fetch_optional_json(
        "get_kpi_metrics",
        baseline_session_id,
        lambda: get_kpi_metrics_tool.get_kpi_metrics(
            baseline_session_id, bypass_cache=bypass_cache
        ),
    )
    opt_kpi = _fetch_optional_json(
        "get_kpi_metrics",
        optimized_session_id,
        lambda: get_kpi_metrics_tool.get_kpi_metrics(
            optimized_session_id, bypass_cache=bypass_cache
        ),
    )
    base_hlo_stats = _fetch_optional_json(
        "get_hlo_stats",
        baseline_session_id,
        lambda: get_hlo_stats_tool.get_hlo_stats(
            baseline_session_id, bypass_cache=bypass_cache
        ),
    )
    opt_hlo_stats = _fetch_optional_json(
        "get_hlo_stats",
        optimized_session_id,
        lambda: get_hlo_stats_tool.get_hlo_stats(
            optimized_session_id, bypass_cache=bypass_cache
        ),
    )

    base_total_us = float(base_summary.get("total_device_duration_us", 0.0))
    opt_total_us = float(opt_summary.get("total_device_duration_us", 0.0))
    delta_total_us = round(opt_total_us - base_total_us, 4)
    delta_total_pct = (
        round((delta_total_us / base_total_us) * 100.0, 2)
        if base_total_us > 0
        else 0.0
    )

    base_kernels = {
        r.get("kernel_name", ""): float(r.get("total_duration_us", 0.0))
        for r in base_summary.get("kernel_records", [])
        if r.get("kernel_name")
    }
    opt_kernels = {
        r.get("kernel_name", ""): float(r.get("total_duration_us", 0.0))
        for r in opt_summary.get("kernel_records", [])
        if r.get("kernel_name")
    }

    all_kernel_names = sorted(
        set(base_kernels.keys()).union(set(opt_kernels.keys())),
        key=lambda k: abs(opt_kernels.get(k, 0.0) - base_kernels.get(k, 0.0)),
        reverse=True,
    )

    kernel_diffs = []
    md_lines = [
        f"# Session Diff: `{baseline_session_id}` vs `{optimized_session_id}`",
        "",
        f"- **Baseline Total Device Time**: {base_total_us:.2f} us",
        f"- **Optimized Total Device Time**: {opt_total_us:.2f} us",
        f"- **Delta**: {delta_total_us:+.2f} us ({delta_total_pct:+.2f}%)",
        "",
        "| Kernel | Baseline (us) | Optimized (us) | Delta (us) | Delta (%) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for kernel_name in all_kernel_names[:20]:
      baseline_us = base_kernels.get(kernel_name, 0.0)
      optimized_us = opt_kernels.get(kernel_name, 0.0)
      delta_us = round(optimized_us - baseline_us, 4)
      delta_pct = (
          round((delta_us / baseline_us) * 100.0, 2) if baseline_us > 0 else 0.0
      )
      kernel_diffs.append({
          "kernel_name": kernel_name,
          "baseline_duration_us": round(baseline_us, 4),
          "optimized_duration_us": round(optimized_us, 4),
          "delta_duration_us": delta_us,
          "delta_percent": delta_pct,
      })
      md_lines.append(
          f"| `{kernel_name}` | {baseline_us:.2f} | {optimized_us:.2f} |"
          f" {delta_us:+.2f} | {delta_pct:+.2f}% |"
      )

    result: dict[str, Any] = {
        "baseline_session_id": str(baseline_session_id),
        "optimized_session_id": str(optimized_session_id),
        "baseline_total_device_duration_us": round(base_total_us, 4),
        "optimized_total_device_duration_us": round(opt_total_us, 4),
        "total_device_duration_delta_us": delta_total_us,
        "total_device_duration_delta_pct": delta_total_pct,
        "kernel_diffs": kernel_diffs,
        "kpi_metrics": {
            "baseline": base_kpi if isinstance(base_kpi, dict) else {},
            "optimized": opt_kpi if isinstance(opt_kpi, dict) else {},
        },
        "hlo_stats": {
            "baseline": (
                base_hlo_stats if isinstance(base_hlo_stats, dict) else {}
            ),
            "optimized": (
                opt_hlo_stats if isinstance(opt_hlo_stats, dict) else {}
            ),
        },
        "markdown_summary": "\n".join(md_lines),
    }
    return json.dumps(result, indent=2)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception(
        "Error diffing sessions %s and %s",
        baseline_session_id,
        optimized_session_id,
    )
    raise RuntimeError(f"Failed to diff sessions: {e}") from e
