"""Orchestration for the XLoop Engine.

The scheduling model is simple by design. One task is one `(item, config)`
pair. Tasks run concurrently up to the number of accelerators the chip broker
can lease, each in its own process pinned to one chip. There is no work
stealing and no cross-host distribution: on a single TPU VM the parallelism
ceiling is the chip count, and anything more elaborate would add failure modes
without adding throughput.
"""

from __future__ import annotations

import collections.abc
import concurrent.futures
import dataclasses
import json
import logging
import os
import shlex
import subprocess
import sys
from typing import Any

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import chip_broker
from xprof.xloop import input_utils
from xprof.xloop import summarize_results_lib
from xprof.xloop import xloop_worker_lib

_Sequence = collections.abc.Sequence

# Overrides the command used to launch a worker. The default invokes this
# package's worker entry point with the running interpreter, which is right
# for a pip install but not for every packaging.
_WORKER_CMD_ENV = "XLOOP_WORKER_CMD"

# Named rather than imported: importing the entry point would register its
# required flags in this process.
_WORKER_MODULE = f"{__package__}.xloop_worker_main"

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class XLoopConfig:
  """Configuration for one engine run.

  Attributes:
    output_dir: Root for per-item artifact directories and results.
    run_reference_code: Execute the reference implementation.
    run_custom_code: Execute the implementation under test.
    run_optimized_code: Execute the optimized implementation.
    num_executions: Timed iterations per implementation.
    num_warmups: Untimed iterations per implementation.
    seed: Input-generation seed, shared by all implementations.
    tpu_topology: Requested accelerator shape, e.g. `vlp=1x1`. Only the chip
      count is read here.
    max_local_workers: Cap on concurrent workers. `None` means the number of
      chips the broker can lease.
    first_config_only: Run only the first input-generation config of each item.
    detailed_summary: Include per-sample detail in the rendered summary.
    verbose: Propagate worker stdout and stderr to this process.
    use_chip_broker: Lease a chip per worker. Disable only when something else
      already owns device assignment.
    in_process: Run tasks in this process instead of spawning workers. Removes
      isolation and forbids concurrency; intended for tests and for debugging a
      single kernel under a debugger.
    lease_timeout_sec: How long a worker waits for a free chip.
    dump_flags: Compiler dump flags forwarded verbatim to the worker, so the
      documented dump recipes reach the compiler that actually runs.
  """

  output_dir: str
  run_reference_code: bool = True
  run_custom_code: bool = True
  run_optimized_code: bool = True
  num_executions: int = 10
  num_warmups: int = 3
  seed: int = 0
  tpu_topology: str = ""
  max_local_workers: int | None = None
  first_config_only: bool = False
  detailed_summary: bool = False
  verbose: bool = False
  use_chip_broker: bool = True
  in_process: bool = False
  lease_timeout_sec: float = 1800.0
  dump_flags: tuple[str, ...] = ()

  def roles(self) -> tuple[str, ...]:
    """Implementation roles enabled by this configuration."""
    enabled = {
        "reference": self.run_reference_code,
        "custom": self.run_custom_code,
        "optimized": self.run_optimized_code,
    }
    return tuple(r for r in input_utils.ROLES if enabled[r])


@dataclasses.dataclass(frozen=True)
class _Task:
  """One `(item, input-generation config)` pair to execute."""

  item_index: int
  item: xloop_item_pb2.XLoopItem
  config_name: str
  output_dir: str


@dataclasses.dataclass(frozen=True)
class _Outcome:
  """What a task produced."""

  task: _Task
  result: xloop_item_pb2.ExecutionResult


def item_label(item: xloop_item_pb2.XLoopItem, index: int) -> str:
  """Names the directory an item's artifacts go in.

  Args:
    item: The item.
    index: Its position in the input file, used only when it has no name or id,
      so that unnamed items still get distinct directories.

  Returns:
    A directory-name component.
  """
  return item.name or item.id or f"item_{index}"


def profile_dir(
    output_dir: str, label: str, config_name: str, role: str
) -> str:
  """Locates the trace directory for one role of one config.

  The single definition of this layout: the worker writes here and the CLI
  reports from here, so they cannot drift apart.

  Args:
    output_dir: Run root, as passed to `run_execution`.
    label: Item label from `item_label`.
    config_name: Input-generation config name.
    role: One of `input_utils.ROLES`.

  Returns:
    The directory path, whether or not it exists.
  """
  return os.path.join(output_dir, label, config_name, f"{role}_profile")


def _chip_count_from_topology(tpu_topology: str) -> int | None:
  """Reads a chip count out of a topology string such as `vlp=2x4`.

  Args:
    tpu_topology: Topology string, possibly with a platform prefix.

  Returns:
    The product of the dimensions, or None if the string is not a topology.
  """
  if not tpu_topology:
    return None
  shape = tpu_topology.split("=", 1)[-1].split(":", 1)[-1]
  try:
    dims = [int(part) for part in shape.split("x")]
  except ValueError:
    return None
  if not dims:
    return None
  product = 1
  for dim in dims:
    product *= dim
  return product or None


def _worker_command() -> list[str]:
  """Returns the argv prefix that launches a worker process."""
  override = os.environ.get(_WORKER_CMD_ENV, "").strip()
  if override:
    return shlex.split(override)
  return [sys.executable, "-m", _WORKER_MODULE]


def _read_json(path: str) -> dict[str, Any]:
  try:
    with open(path) as f:
      loaded = json.load(f)
  except (OSError, json.JSONDecodeError):
    return {}
  return loaded if isinstance(loaded, dict) else {}


def _failed_result(message: str) -> xloop_item_pb2.ExecutionResult:
  return xloop_item_pb2.ExecutionResult(
      execution_status=xloop_item_pb2.CODE_STATUS_EXECUTE_ERROR,
      error_message=message,
  )


def _run_in_process(task: _Task, config: XLoopConfig) -> _Outcome:
  """Executes a task in this process, without isolation."""
  item_path = os.path.join(task.output_dir, "item.json")
  input_utils.save_output_items([task.item], item_path)
  try:
    results = xloop_worker_lib.run_worker(
        item_path=item_path,
        output_dir=task.output_dir,
        config_name=task.config_name,
        roles=config.roles(),
        num_warmups=config.num_warmups,
        num_executions=config.num_executions,
        seed=config.seed,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.exception("in-process task failed: %s", task.config_name)
    return _Outcome(task, _failed_result(str(e)))
  return _Outcome(
      task,
      summarize_results_lib.build_execution_result(results, config.roles()),
  )


def _run_in_subprocess(
    task: _Task, config: XLoopConfig, env: dict[str, str | None]
) -> _Outcome:
  """Executes a task in a dedicated worker process."""
  item_path = os.path.join(task.output_dir, "item.json")
  input_utils.save_output_items([task.item], item_path)

  argv = _worker_command() + [
      f"--item_path={item_path}",
      f"--output_dir={task.output_dir}",
      f"--config_name={task.config_name}",
      f"--roles={','.join(config.roles())}",
      f"--num_warmups={config.num_warmups}",
      f"--num_executions={config.num_executions}",
      f"--seed={config.seed}",
  ]
  argv.extend(config.dump_flags)

  # A None value means "remove this variable". The broker relies on it to
  # clear device-selection variables the caller may have exported, which
  # would otherwise outrank the one it sets and send the worker to a chip it
  # was not leased.
  worker_env = dict(os.environ)
  for name, value in env.items():
    if value is None:
      worker_env.pop(name, None)
    else:
      worker_env[name] = value

  logger.info("launching worker for %s", task.output_dir)
  completed = subprocess.run(
      argv,
      env=worker_env,
      capture_output=not config.verbose,
      text=True,
      check=False,
  )

  status = _read_json(
      os.path.join(task.output_dir, xloop_worker_lib.STATUS_FILENAME)
  )
  if completed.returncode != 0 or status.get("status") != "EXECUTE_SUCCESS":
    message = status.get("error_message") or (
        f"worker exited with code {completed.returncode}"
    )
    if not config.verbose and completed.stderr:
      # Without this the failure is a bare exit code, which is not actionable.
      message = f"{message}\n{completed.stderr[-4000:]}"
    return _Outcome(task, _failed_result(message))

  results = _read_json(
      os.path.join(task.output_dir, xloop_worker_lib.RESULTS_FILENAME)
  )
  if not results:
    return _Outcome(task, _failed_result("worker produced no results.json"))
  return _Outcome(
      task,
      summarize_results_lib.build_execution_result(results, config.roles()),
  )


def _build_tasks(
    items: _Sequence[xloop_item_pb2.XLoopItem], config: XLoopConfig
) -> list[_Task]:
  """Expands items into one task per input-generation config."""
  tasks: list[_Task] = []
  for index, item in enumerate(items):
    config_names = sorted(item.input_generation_configs)
    if not config_names:
      logger.warning(
          "skipping item %r: no input_generation_configs",
          item.name or item.id,
      )
      continue
    if config.first_config_only:
      config_names = config_names[:1]
    label = item_label(item, index)
    for config_name in config_names:
      tasks.append(
          _Task(
              item_index=index,
              item=item,
              config_name=config_name,
              output_dir=os.path.join(config.output_dir, label, config_name),
          )
      )
  return tasks


def run_execution(
    items: _Sequence[xloop_item_pb2.XLoopItem], config: XLoopConfig
) -> list[xloop_item_pb2.XLoopItem]:
  """Executes every item and returns copies populated with results.

  Args:
    items: Items to execute. Not mutated.
    config: Run configuration.

  Returns:
    Copies of `items`, each with `execution_results` and `execution_report`
    filled in. Items that could not be executed carry an error status rather
    than being dropped, so a partial run is still reportable.
  """

  out = [xloop_item_pb2.XLoopItem() for _ in items]
  for copy, original in zip(out, items):
    copy.CopyFrom(original)

  tasks = _build_tasks(out, config)
  if not tasks:
    return out

  for task in tasks:
    os.makedirs(task.output_dir, exist_ok=True)

  broker: chip_broker.ChipBroker | None = None
  if config.use_chip_broker and not config.in_process:
    requested = config.max_local_workers or _chip_count_from_topology(
        config.tpu_topology
    )
    try:
      broker = chip_broker.ChipBroker(num_chips=requested)
      # A previous run killed mid-flight would otherwise hold capacity
      # forever, since a dead process releases nothing.
      reaped = broker.reap_all()
      if reaped:
        logger.info("reaped %d stale lease file(s)", reaped)
    except chip_broker.ChipBrokerError as e:
      logger.info("running without a chip broker: %s", e)

  if config.in_process:
    outcomes = [_run_in_process(task, config) for task in tasks]
  else:
    outcomes = _run_concurrently(tasks, config, broker)

  for outcome in outcomes:
    item = out[outcome.task.item_index]
    item.execution_results[outcome.task.config_name].CopyFrom(outcome.result)

  for item in out:
    if item.execution_results:
      item.execution_report.CopyFrom(
          summarize_results_lib.aggregate_report(item)
      )
  return out


def _run_concurrently(
    tasks: _Sequence[_Task],
    config: XLoopConfig,
    broker: chip_broker.ChipBroker | None,
) -> list[_Outcome]:
  """Runs tasks across the available chips."""
  if broker is None:
    max_workers = config.max_local_workers or 1
  else:
    max_workers = min(
        broker.capacity, config.max_local_workers or broker.capacity
    )
  max_workers = max(1, min(max_workers, len(tasks)))

  def run_one(task: _Task) -> _Outcome:
    if broker is None:
      return _run_in_subprocess(task, config, env={})

    def on_wait(position: int, capacity: int) -> None:
      # Surfaced so a queued run is distinguishable from a hung one.
      logger.info(
          "waiting for a chip: position %d of %d in use", position, capacity
      )

    try:
      with broker.lease(
          timeout_sec=config.lease_timeout_sec, on_wait=on_wait
      ) as claim:
        return _run_in_subprocess(task, config, env=claim.env())
    except chip_broker.ChipBrokerError as e:
      return _Outcome(task, _failed_result(str(e)))

  if max_workers == 1:
    return [run_one(task) for task in tasks]

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    return list(pool.map(run_one, tasks))
