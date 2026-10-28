"""The `xloop` command-line interface.

A separate binary from `xprof`, because the two have different jobs and
different dependencies: `xprof` reads profiles that some other process
produced and must stay installable without a JAX toolchain, while `xloop`
runs the workload itself and needs JAX and an attached accelerator.

The two compose through the filesystem. `xloop run` writes ordinary XProf
log directories, so the analyses apply directly to its output:

    xloop run --item_path=kernels.json --output_dir=/tmp/run
    xprof get_kernel_stats --logdir=/tmp/run/<item>/<config>/custom_profile

The functions below return plain JSON-serializable structures and are also
the programmatic API; `main` only adds argument parsing and error
formatting.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import logging
import os
import sys
import traceback
from typing import Any

import fire

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import chip_broker
from xprof.xloop import input_utils
from xprof.xloop import summarize_results_lib
from xprof.xloop import xloop_lib

_Sequence = collections.abc.Sequence


def xloop_run(
    item_path: str,
    output_dir: str,
    num_executions: int = 10,
    num_warmups: int = 3,
    seed: int = 0,
    roles: str = "reference,custom,optimized",
    tpu_topology: str = "",
    max_local_workers: int | None = None,
    first_config_only: bool = False,
    detailed_summary: bool = False,
    use_chip_broker: bool = True,
    in_process: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
  """Benchmarks JAX functions or Pallas kernels and writes profile traces.

  Each implementation is compiled, warmed up and timed in its own process,
  pinned to one accelerator. Every implementation sees byte-identical inputs,
  so outputs are directly comparable; results include a correctness verdict
  against the reference.

  The trace directory each implementation writes is a valid `xprof` log
  directory. To go deeper, point the analysis tools at it:
  `get_kernel_stats` for per-kernel device time, `get_llo_analysis` for
  bundle-level utilization and stalls, `verify_numerical_parity` for a
  ULP-level correctness audit.

  Args:
    item_path: Path to a `.json`, `.jsonl`, `.txtpb` or `.pb` file holding one
      or more `XLoopItem`s.
    output_dir: Root directory for traces, results and the output items file.
    num_executions: Timed iterations per implementation.
    num_warmups: Untimed iterations run first, to force compilation and settle
      caches.
    seed: Input-generation seed. Fixed by default so runs are comparable.
    roles: Comma-separated implementations to run, from `reference`, `custom`,
      `optimized`.
    tpu_topology: Requested accelerator shape, e.g. `vlp=2x4`. Only the chip
      count is used.
    max_local_workers: Cap on concurrent workers. Defaults to the number of
      leasable chips.
    first_config_only: Run only each item's first input-generation config.
    detailed_summary: Include per-sample timings in the returned summary.
    use_chip_broker: Lease a chip per worker. Disable only if something else
      already assigns devices.
    in_process: Run in this process without isolation. Forces serial execution;
      intended for debugging a single kernel.
    verbose: Stream worker output instead of capturing it.

  Returns:
    A dictionary with the rendered `summary` table, a machine-readable
    `items` list, and `output_items_path`.

  Raises:
    ValueError: If `roles` names something that is not an implementation role.
  """
  selected = [r.strip() for r in roles.split(",") if r.strip()]
  unknown = [r for r in selected if r not in input_utils.ROLES]
  if unknown:
    raise ValueError(
        f"unknown role(s) {unknown}; expected any of {list(input_utils.ROLES)}"
    )

  items = input_utils.load_input_items(item_path)
  if not items:
    return {"summary": "", "items": [], "output_items_path": ""}

  config = xloop_lib.XLoopConfig(
      output_dir=output_dir,
      run_reference_code="reference" in selected,
      run_custom_code="custom" in selected,
      run_optimized_code="optimized" in selected,
      num_executions=num_executions,
      num_warmups=num_warmups,
      seed=seed,
      tpu_topology=tpu_topology,
      max_local_workers=max_local_workers,
      first_config_only=first_config_only,
      detailed_summary=detailed_summary,
      use_chip_broker=use_chip_broker,
      in_process=in_process,
      verbose=verbose,
  )

  executed = xloop_lib.run_execution(items, config)

  output_items_path = os.path.join(output_dir, "items.json")
  input_utils.save_output_items(executed, output_items_path)

  payload: dict[str, Any] = {
      "summary": summarize_results_lib.format_summary(executed),
      "items": [
          _item_summary(output_dir, item, i) for i, item in enumerate(executed)
      ],
      "output_items_path": output_items_path,
  }
  if detailed_summary:
    payload["detail"] = "\n\n".join(
        summarize_results_lib.format_detail(item) for item in executed
    )
  return payload


def xloop_summarize(
    item_path: str, detailed_summary: bool = False
) -> dict[str, Any]:
  """Re-renders the summary for an already-executed items file.

  Args:
    item_path: Path to an items file carrying execution results, typically the
      `items.json` written by `xloop_run`.
    detailed_summary: Include per-sample timings.

  Returns:
    A dictionary with the rendered `summary`, optional `detail`, and a
    machine-readable `items` list.
  """
  items = input_utils.load_input_items(item_path)
  # `xloop_run` writes items.json at the run root, so the traces sit beside it.
  output_dir = os.path.dirname(os.path.abspath(item_path))
  payload: dict[str, Any] = {
      "summary": summarize_results_lib.format_summary(items),
      "items": [
          _item_summary(output_dir, item, i) for i, item in enumerate(items)
      ],
  }
  if detailed_summary:
    payload["detail"] = "\n\n".join(
        summarize_results_lib.format_detail(item) for item in items
    )
  return payload


def xloop_chips() -> dict[str, Any]:
  """Reports the accelerators on this host and the current lease state.

  Reads the PCI bus rather than asking JAX, because initializing JAX would
  claim every visible chip.

  Returns:
    A dictionary describing discovered chips, the lease directory, and which
    chips are currently leased.
  """
  chips = chip_broker.discover_chips()
  lease_dir = chip_broker.default_lease_dir()
  # A lock file outlives its lease by design -- see `ChipBroker._try_claim` --
  # so its existence says nothing. Whether the lock is *held* is the question,
  # and the only way to ask is to try to take it.
  leased = sorted(
      chip.index
      for chip in chips
      if chip_broker.is_chip_leased(lease_dir, chip.index)
  )
  return {
      "num_chips": len(chips),
      "chips": [dataclasses.asdict(c) for c in chips],
      "lease_dir": lease_dir,
      "leased_chip_indices": leased,
      "available": max(len(chips) - len(leased), 0),
  }


def _item_summary(
    output_dir: str, item: xloop_item_pb2.XLoopItem, index: int
) -> dict[str, Any]:
  """Condenses one item's results into a JSON-friendly record."""
  report = item.execution_report
  errors = {
      name: result.error_message
      for name, result in item.execution_results.items()
      if result.error_message
  }
  return {
      "name": item.name,
      "id": item.id,
      "status": xloop_item_pb2.CodeStatus.Name(report.execution_status),
      "custom_outputs_match_reference": report.custom_outputs_match_reference,
      "optimized_outputs_match_reference": (
          report.optimized_outputs_match_reference
      ),
      "gmean_speedup_over_reference": report.gmean_speedup_over_reference,
      "gmean_speedup_over_optimized": report.gmean_speedup_over_optimized,
      "gmean_speedup_optimized_over_reference": (
          report.gmean_speedup_optimized_over_reference
      ),
      "trace_dirs": _trace_dirs(output_dir, item, index),
      "errors": errors,
  }


def _trace_dirs(
    output_dir: str, item: xloop_item_pb2.XLoopItem, index: int
) -> dict[str, str]:
  """Maps `<config>/<role>` to the trace directory the engine wrote.

  Reconstructed from the run layout rather than read back from the item,
  because the directory is what the downstream analyses are pointed at and
  it must be reported even for roles the item does not record a result for.

  Args:
    output_dir: Run root, as passed to `xloop_run`.
    item: Executed item.
    index: The item's position in the input file.

  Returns:
    Mapping of `<config>/<role>` to directory, for those that exist on disk.
  """
  label = xloop_lib.item_label(item, index)
  out: dict[str, str] = {}
  for config_name in item.execution_results:
    for role in input_utils.ROLES:
      candidate = xloop_lib.profile_dir(output_dir, label, config_name, role)
      if os.path.isdir(candidate):
        out[f"{config_name}/{role}"] = candidate
  return out


class XLoopCli:
  """Subcommands of the `xloop` binary."""

  def __init__(self):
    self.run = xloop_run
    self.summarize = xloop_summarize
    self.chips = xloop_chips


def _emit_error(
    reason: str,
    message: str,
    exit_code: int,
    traceback_str: str | None = None,
) -> None:
  """Emits structured JSON on stdout and a human-readable line on stderr.

  Mirrors `xprof_cli`'s error contract -- same keys, same exit codes -- so a
  caller driving both binaries can parse failures the same way.

  Args:
    reason: Stable machine-readable error category.
    message: Human-readable description.
    exit_code: Process exit status.
    traceback_str: Formatted traceback, for unexpected failures only.
  """
  payload = {"status": "ERROR", "reason": reason, "error": message}
  if traceback_str:
    payload["traceback"] = traceback_str
  sys.stdout.write(json.dumps(payload, indent=2) + "\n")
  sys.stderr.write(f"{reason}: {message}\n")
  if traceback_str:
    sys.stderr.write(f"\n{traceback_str}\n")
  sys.exit(exit_code)


def main(argv=None) -> None:
  """Entry point for the `xloop` binary."""
  if argv is None:
    argv = sys.argv
  try:
    fire.Fire(XLoopCli(), command=argv[1:] if argv else None, name="xloop")
  except (fire.core.FireError, TypeError) as e:
    _emit_error("USAGE_ERROR", str(e), 2)
  except OSError as e:
    _emit_error("PATH_ERROR", str(e), 3)
  except ValueError as e:
    _emit_error("INVALID_VALUE", str(e), 4)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Unhandled defect in xloop_cli")
    _emit_error(
        "INTERNAL_ERROR",
        str(e),
        1,
        traceback_str=traceback.format_exc().strip(),
    )


if __name__ == "__main__":
  main(sys.argv)
