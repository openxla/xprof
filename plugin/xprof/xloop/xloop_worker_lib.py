"""Worker that executes one benchmark item in an isolated process.

Why a subprocess at all
-----------------------
Three things require process isolation even on a single host:

  * **Device pinning.** `TPU_VISIBLE_DEVICE_PATHS` is read by libtpu at
    initialization and cannot be changed afterwards, so one process can serve
    exactly one chip lease.
  * **Clean JAX state.** Compilation caches, donated buffers and live
    `jax.Array`s from a previous item would otherwise perturb the next one.
  * **Crash containment.** A kernel that segfaults libtpu takes down only its
    own worker.
"""

from __future__ import annotations

import collections.abc
import importlib.util
import json
import logging
import os
import signal
import sys
import traceback
import types
from typing import Any

from xprof.protobuf import xloop_item_pb2
from xprof.xloop import input_utils
from xprof.xloop import trace_timing_lib

_Sequence = collections.abc.Sequence

# Written by the worker on completion, successful or not, so the orchestrator
# can distinguish "failed" from "never started".
RESULTS_FILENAME = "results.json"
STATUS_FILENAME = "status.json"

logger = logging.getLogger(__name__)


class WorkerError(RuntimeError):
  """Raised when an item cannot be executed at all."""


def _write_module(work_dir: str, module_name: str, source: str) -> str:
  """Writes `source` to the file implied by a dotted module name.

  Args:
    work_dir: Root of the importable tree.
    module_name: Dotted module name, e.g. `pkg.sub.mod`.
    source: Module source text.

  Returns:
    Path to the written file.
  """
  parts = module_name.split(".")
  directory = os.path.join(work_dir, *parts[:-1])
  os.makedirs(directory, exist_ok=True)
  # Make every intermediate directory a package.
  for depth in range(len(parts) - 1):
    init_path = os.path.join(work_dir, *parts[: depth + 1], "__init__.py")
    if not os.path.exists(init_path):
      with open(init_path, "w"):
        pass
  path = os.path.join(directory, f"{parts[-1]}.py")
  with open(path, "w") as f:
    f.write(source)
  return path


def _load_module_from_path(path: str, module_name: str) -> types.ModuleType:
  """Imports a module from an explicit file path.

  Args:
    path: File to load.
    module_name: Name to register it under.

  Returns:
    The imported module.

  Raises:
    WorkerError: If the module cannot be loaded or executed.
  """
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise WorkerError(f"cannot load {module_name} from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  try:
    spec.loader.exec_module(module)
  except Exception as e:
    raise WorkerError(f"{module_name} raised during import: {e}") from e
  return module


def _materialize_role(
    item: xloop_item_pb2.XLoopItem, role: str, work_dir: str
) -> types.ModuleType:
  """Writes a role's code plus its auxiliary modules, then imports it."""
  for name, source in input_utils.auxiliary_modules_for_role(
      item, role
  ).items():
    _write_module(work_dir, name, source)
  path = os.path.join(work_dir, f"{role}_code.py")
  with open(path, "w") as f:
    f.write(input_utils.code_for_role(item, role))
  return _load_module_from_path(path, f"xloop_{role}_code")


def _materialize_input_generation(
    config: xloop_item_pb2.InputGenerationConfig, work_dir: str
) -> types.ModuleType:
  """Writes and imports an input-generation config's code."""
  for name, source in config.auxiliary_modules.modules.items():
    _write_module(work_dir, name, source)
  path = os.path.join(work_dir, "input_generation_code.py")
  with open(path, "w") as f:
    f.write(config.input_generation_code)
  return _load_module_from_path(path, "xloop_input_generation_code")


def _required_attr(module: types.ModuleType, name: str) -> Any:
  """Fetches a required module-level factory.

  Args:
    module: Module to read.
    name: Attribute name.

  Returns:
    The attribute's value.

  Raises:
    WorkerError: If the attribute is missing or not callable.
  """
  fn = getattr(module, name, None)
  if fn is None:
    raise WorkerError(
        f"{module.__name__} must define {name}(); see the XLoopItem module"
        " contract"
    )
  if not callable(fn):
    raise WorkerError(f"{module.__name__}.{name} is not callable")
  return fn()


def _optional_float(module: types.ModuleType, name: str) -> float | None:
  fn: Any = getattr(module, name, None)
  if fn is None or not callable(fn):
    return None
  value: Any = fn()
  return None if value is None else float(value)


def select_config(
    item: xloop_item_pb2.XLoopItem, config_name: str
) -> tuple[str, xloop_item_pb2.InputGenerationConfig]:
  """Picks the input-generation config to run.

  Args:
    item: Item being executed.
    config_name: Requested name, or empty for the first in sorted order.

  Returns:
    The chosen `(name, config)`.

  Raises:
    WorkerError: If the item has no configs, or the named one is absent.
  """
  configs = dict(item.input_generation_configs)
  if not configs:
    raise WorkerError(
        f"item {item.name!r} declares no input_generation_configs"
    )
  if not config_name:
    # Sorted rather than arbitrary, so repeated runs pick the same one.
    config_name = sorted(configs)[0]
  if config_name not in configs:
    raise WorkerError(
        f"item {item.name!r} has no input-generation config"
        f" {config_name!r}; available: {sorted(configs)}"
    )
  return config_name, configs[config_name]


def _write_status(output_dir: str, status: dict[str, Any]) -> None:
  os.makedirs(output_dir, exist_ok=True)
  with open(os.path.join(output_dir, STATUS_FILENAME), "w") as f:
    json.dump(status, f, indent=2, sort_keys=True)


def _register_sigterm_handler(output_dir: str) -> None:
  """Records termination by signal, which otherwise leaves no artifact.

  A worker killed by the orchestrator or the OOM killer would be
  indistinguishable from one that never started. Writing a status file on
  SIGTERM lets the orchestrator report the difference.

  Args:
    output_dir: Where to write the status file.
  """

  def handler(signum: int, frame: Any) -> None:
    del frame
    _write_status(
        output_dir,
        {
            "status": "EXECUTE_ERROR",
            "error_message": f"worker terminated by signal {signum}",
        },
    )
    # Restore default disposition and re-raise so the exit status is accurate.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

  try:
    signal.signal(signal.SIGTERM, handler)
  except ValueError:
    # Not on the main thread; the orchestrator will fall back to exit codes.
    logger.debug("could not install SIGTERM handler")


def run_worker(
    item_path: str,
    output_dir: str,
    config_name: str = "",
    roles: _Sequence[str] = input_utils.ROLES,
    num_warmups: int = 3,
    num_executions: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
  """Executes one item and writes its results.

  Args:
    item_path: Path to a file holding exactly one `XLoopItem`.
    output_dir: Destination for traces, `results.json` and `status.json`.
    config_name: Input-generation config to run. Empty picks the first.
    roles: Implementation roles to execute, filtered to those the item defines.
    num_warmups: Untimed iterations.
    num_executions: Timed iterations.
    seed: Input-generation seed.

  Returns:
    The results dictionary, with output arrays removed.

  Raises:
    WorkerError: If the item cannot be executed.
  """
  _register_sigterm_handler(output_dir)

  items = input_utils.load_input_items(item_path)
  if len(items) != 1:
    raise WorkerError(
        f"expected exactly one item in {item_path}, found {len(items)}"
    )
  item = items[0]

  resolved_config_name, config = select_config(item, config_name)

  requested = [r for r in roles if r in input_utils.ROLES]
  available = input_utils.present_roles(item)
  selected = [r for r in requested if r in available]
  if not selected:
    raise WorkerError(
        f"item {item.name!r} defines no code for any of the requested roles"
        f" {list(roles)}; it defines {list(available)}"
    )

  work_dir = os.path.join(output_dir, "work")
  os.makedirs(work_dir, exist_ok=True)
  # Auxiliary modules are imported by name, so the work directory has to be
  # importable. Prepended so it shadows nothing unintentionally on removal.
  sys.path.insert(0, work_dir)
  try:
    return _execute(
        item=item,
        config=config,
        config_name=resolved_config_name,
        roles=selected,
        work_dir=work_dir,
        output_dir=output_dir,
        num_warmups=num_warmups,
        num_executions=num_executions,
        seed=seed,
    )
  finally:
    try:
      sys.path.remove(work_dir)
    except ValueError:
      pass


def _execute(
    *,
    item: xloop_item_pb2.XLoopItem,
    config: xloop_item_pb2.InputGenerationConfig,
    config_name: str,
    roles: _Sequence[str],
    work_dir: str,
    output_dir: str,
    num_warmups: int,
    num_executions: int,
    seed: int,
) -> dict[str, Any]:
  """Runs the measurement; imports JAX lazily so errors are reportable."""
  # Imported here rather than at module scope: the engine must be importable
  # without JAX so that `xprof`'s CLI registry loads on a JAX-free install.
  from xprof.xloop import eval_jax_lib  # pylint: disable=g-import-not-at-top

  input_module = _materialize_input_generation(config, work_dir)
  input_generator = _required_attr(input_module, "get_input_generator")
  get_init_inputs: Any = getattr(input_module, "get_init_inputs", None)
  init_inputs_factory: Any = None
  if get_init_inputs is not None and callable(get_init_inputs):
    init_inputs_factory = get_init_inputs()

  item_atol, item_rtol = input_utils.tolerances(config)
  atol = item_atol
  if atol is None:
    atol = _optional_float(input_module, "get_atol")
  if atol is None:
    atol = input_utils.DEFAULT_ATOL
  rtol = item_rtol
  if rtol is None:
    rtol = _optional_float(input_module, "get_rtol")
  if rtol is None:
    rtol = input_utils.DEFAULT_RTOL

  funcs: list[tuple[str, Any]] = []
  for role in roles:
    module = _materialize_role(item, role, work_dir)
    model_identifier = _required_attr(module, "get_model_identifier")
    funcs.append((
        role,
        eval_jax_lib.benchmark_func_for_model(
            model_identifier, input_generator, init_inputs_factory
        ),
    ))

  warmup_keys, actual_keys = eval_jax_lib.sample_keys(
      seed, num_warmups, num_executions
  )
  results = eval_jax_lib.profile_funcs(
      funcs,
      input_generator,
      warmup_keys,
      actual_keys,
      list(range(num_warmups)),
      list(range(num_executions)),
      config_name=config_name,
      directory=output_dir,
  )

  # Device-derived per-sample times, which the summarizer prefers over the
  # wall-clock ones. Wall clock around a jitted call is dominated by host
  # dispatch (~160us on a v6e), which swamps any microbenchmark.
  trace_timing_lib.annotate_results(results, roles)

  # Correctness, always against the reference role.
  reference_outputs = results.get(f"{input_utils.REFERENCE_ROLE}_outputs")
  for role in roles:
    if role == input_utils.REFERENCE_ROLE:
      continue
    verdicts = eval_jax_lib.check_equal(
        reference_outputs, results.get(f"{role}_outputs"), rtol=rtol, atol=atol
    )
    results[f"{role}_outputs_match_reference"] = (
        bool(verdicts) if isinstance(verdicts, bool) else all(verdicts)
    )
    results[f"{role}_per_sample_match"] = (
        [] if isinstance(verdicts, bool) else list(verdicts)
    )

  results["config_name"] = config_name
  results["item_name"] = item.name
  results["item_id"] = item.id
  results["atol"] = atol
  results["rtol"] = rtol
  results["num_warmups"] = num_warmups
  results["num_executions"] = num_executions
  results["seed"] = seed

  eval_jax_lib.serialize_results(results, output_dir)
  _write_status(output_dir, {"status": "EXECUTE_SUCCESS", "error_message": ""})
  return {k: v for k, v in results.items() if not k.endswith("_outputs")}


def run_worker_reporting_failures(
    item_path: str,
    output_dir: str,
    config_name: str = "",
    roles: _Sequence[str] = input_utils.ROLES,
    num_warmups: int = 3,
    num_executions: int = 10,
    seed: int = 0,
) -> int:
  """Runs the worker, recording any failure as a status file.

  A worker failure is data, not a crash: the orchestrator needs the message
  and traceback to attribute the failure to the right kernel, and a process
  that dies without writing anything is indistinguishable from one that never
  started.

  Args:
    item_path: Path to the item to execute.
    output_dir: Destination for traces, results and status.
    config_name: Input-generation config to run.
    roles: Implementation roles to execute.
    num_warmups: Untimed iterations.
    num_executions: Timed iterations.
    seed: Input-generation seed.

  Returns:
    A process exit code: 0 on success, 1 on failure.
  """
  try:
    run_worker(
        item_path=item_path,
        output_dir=output_dir,
        config_name=config_name,
        roles=roles,
        num_warmups=num_warmups,
        num_executions=num_executions,
        seed=seed,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    _write_status(
        output_dir,
        {
            "status": "EXECUTE_ERROR",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        },
    )
    logger.error("worker failed: %s", e)
    return 1
  return 0
