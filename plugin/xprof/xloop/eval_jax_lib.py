# Copyright 2026 The XProf Authors. All Rights Reserved.
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
"""JAX/Pallas measurement harness for the XLoop Engine.

Compiles a candidate callable, runs untimed warmup iterations, executes timed
samples inside `jax.profiler.trace` regions with a separator op between
samples so executions can be matched on the device timeline, and compares
per-sample outputs against the reference implementation. When several
implementations run, their timed samples run in alternating blocks, so a drift
of the device over time does not favor one of them.

Requires `jax`. The engine imports this module lazily so that `xprof` itself
keeps working without a JAX install.
"""

from __future__ import annotations

import collections.abc
import functools
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import shlex
import socket
import time
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

_Callable = collections.abc.Callable
_Sequence = collections.abc.Sequence

BenchmarkFunc = _Callable[..., Any]
InputGenerator = _Callable[..., tuple[_Sequence[Any], dict[str, Any]]]
# The positional and keyword inputs of one sample.
_SampleInputs = tuple[_Sequence[Any], dict[str, Any]]
ModelIdentifier = Any
RandomKey = Any

# Comparison tolerances used when the caller supplies none.
_DEFAULT_RTOL = 1e-5
_DEFAULT_ATOL = 1e-5

logger = logging.getLogger(__name__)

# Timed blocks per role in `profile_funcs`. Two give the order A B B A, which
# cancels a drift that is linear in time.
DEFAULT_NUM_BLOCKS = 2

# Untimed warmup samples a role runs again before each of its blocks after the
# first. See `profile_funcs`.
_REWARMUP_SAMPLES = 2

# The first line of lowered StableHLO text: `module @jit_f attributes {...}`.
# A name with characters outside the bare identifier set is quoted.
_MODULE_NAME_RE = re.compile(r'^module @(?:"([^"]+)"|([\w.$-]+))', re.MULTILINE)


def _try_import(module_name: str) -> Any:
  """Imports a module by name, returning None if it is unavailable."""
  try:
    return importlib.import_module(module_name)
  except ImportError:
    return None


def _profiler_options() -> Any:
  """Returns profiler options with host tracing off, or None if unsupported.

  Host tracing is disabled because the engine measures device work; host events
  add substantial trace volume and are not read by any consumer here.

  Returns:
    A `jax.profiler.ProfileOptions`, or None on JAX versions that predate it.
  """
  make_options = getattr(jax.profiler, "ProfileOptions", None)
  if make_options is None:
    return None
  options = make_options()
  options.host_tracer_level = 0
  return options


def trace_context(trace_dir: str) -> Any:
  """Opens a profiling trace writing to `trace_dir`.

  The resulting layout is `<trace_dir>/plugins/profile/<run>/*.xplane.pb`,
  which is exactly what `xprof --logdir` expects, so a trace directory is
  directly loadable without post-processing.

  Args:
    trace_dir: Destination directory. Created if absent.

  Returns:
    A context manager that traces for its duration.
  """
  os.makedirs(trace_dir, exist_ok=True)
  options = _profiler_options()
  if options is None:
    return jax.profiler.trace(trace_dir)
  return jax.profiler.trace(trace_dir, profiler_options=options)


@jax.jit
def dummy_op() -> jax.Array:
  """A small, cheap device op used to separate samples in the trace.

  Back-to-back executions of the same computation can be merged into a single
  trace event, which destroys per-sample timing. Interposing an unrelated op
  forces a boundary.

  Returns:
    A scalar; the value is not used.
  """
  return jnp.sum(
      jax.random.normal(jax.random.key(0), (1024, 1024), jnp.float32)
  )


class ProfileResult(NamedTuple):
  """Outcome of profiling one implementation.

  Attributes:
    outputs: Per-sample outputs, retained for the correctness check.
    trace_dir: Directory holding the profile.
    compilation_sec: Wall time of the first (compiling) call.
    warmup_sec: Wall time of all warmup samples.
    active_sec: Wall time of all timed samples.
    per_sample_sec: Wall time of each timed sample individually. The summarizer
      needs the distribution, not just the total: it sizes its
      faster/slower/no-change threshold from the spread, so an aggregate alone
      would collapse that threshold to its floor.
    module_name: Name of the XLA module that runs the implementation, as it
      appears on the trace's `XLA Modules` line without the program id, e.g.
      `jit__matmul`. Empty when it is not known, for example for a callable that
      manages its own jit boundary.
  """

  outputs: list[Any]
  trace_dir: str
  compilation_sec: float
  warmup_sec: float
  active_sec: float
  per_sample_sec: list[float]
  module_name: str = ""


def _module_name_from_lowered(lowered: Any) -> str:
  """Returns the module name of a `jax.stages.Lowered`, or "" if unknown.

  XLA names the compiled module after the StableHLO module, and the profiler
  writes that name on the `XLA Modules` line. So the name in the lowered text
  is the name the trace reader must look for.

  Args:
    lowered: The result of `jax.jit(f).lower(...)`.

  Returns:
    The module name, e.g. `jit__matmul`.
  """
  try:
    text = lowered.as_text()
  except Exception:  # pylint: disable=broad-exception-caught
    # Diagnostic only: an unknown name falls back to a choice by the reader.
    logger.warning("Cannot read the lowered module text.", exc_info=True)
    return ""
  match = _MODULE_NAME_RE.search(text)
  if match is None:
    return ""
  return match.group(1) or match.group(2)


def get_static_argnames(fn: Any) -> _Sequence[str]:
  """Returns the keyword arguments `fn` wants treated as compile-time constants.

  A kernel marks these by setting a `static_argnames` attribute on the callable
  it returns from `get_model_identifier()`.

  Args:
    fn: Candidate benchmark callable.

  Returns:
    Static argument names, empty if the callable declares none.
  """
  names = getattr(fn, "static_argnames", ())
  if isinstance(names, str):
    return (names,)
  try:
    return tuple(names)
  except TypeError:
    return ()


def _accepts_kwarg(f: Any, kwarg_name: str) -> bool:
  """Whether `f` accepts `kwarg_name`, seeing through partials and instances."""
  real_callable: Any = f
  if not inspect.isroutine(real_callable) and not inspect.isclass(
      real_callable
  ):
    # A callable instance: inspect its bound __call__, which carries the real
    # signature. Plain functions and classes are already introspectable.
    bound_call = getattr(real_callable, "__call__", None)
    if bound_call is not None:
      real_callable = bound_call

  while isinstance(real_callable, functools.partial):
    real_callable = real_callable.func

  if not callable(real_callable):
    return False
  try:
    sig = inspect.signature(real_callable)
  except (ValueError, TypeError):
    return False
  return kwarg_name in sig.parameters


def _resolve_memory_space(name: str) -> Any:
  """Resolves a memory-space name such as 'hbm' or 'vmem' to a JAX object.

  Imported dynamically: Pallas is an optional JAX submodule, and the memory
  space enum has been spelled both `MemorySpace` and `TPUMemorySpace` across
  releases.

  Args:
    name: Memory space name, case-insensitive.

  Returns:
    The resolved memory-space object, or None if this JAX build does not
    expose memory spaces.
  """
  pltpu = _try_import("jax.experimental.pallas.tpu")
  if pltpu is None:
    return None
  space = getattr(pltpu, "MemorySpace", None) or getattr(
      pltpu, "TPUMemorySpace", None
  )
  if space is None:
    return None
  return getattr(space, name.upper(), None)


def _constrain_to_memory_spaces(
    values: _Sequence[Any], memory_spaces: _Sequence[str] | None
) -> list[Any]:
  """Best-effort placement of `values` into the named memory spaces.

  Args:
    values: Arrays to place.
    memory_spaces: Parallel sequence of memory-space names. Entries that are
      empty or unresolvable leave the corresponding value untouched.

  Returns:
    The values, placed where possible.
  """
  if not memory_spaces:
    return list(values)
  out = list(values)
  for i, space_name in enumerate(memory_spaces):
    if i >= len(out) or not space_name:
      continue
    space = _resolve_memory_space(space_name)
    if space is None:
      logger.warning(
          "memory space %r is not available in this JAX build; leaving"
          " argument %d unconstrained",
          space_name,
          i,
      )
      continue
    try:
      out[i] = jax.device_put(out[i], space)
    except (TypeError, ValueError) as e:
      logger.warning("could not place argument %d in %s: %s", i, space_name, e)
  return out


def _jit_unless_skipped(func: BenchmarkFunc, takes_config_name: bool) -> Any:
  """Returns `func` under `jax.jit`, or None if `func` sets `skip_jit`.

  Args:
    func: The benchmark callable.
    takes_config_name: Whether `func` receives the config name. The name is a
      string, so it cannot be traced and has to be a compile-time constant.
      Kernels should not have to declare this themselves.
  """
  if getattr(func, "skip_jit", False):
    # The kernel manages its own jit boundary.
    return None
  static_argnames = tuple(get_static_argnames(func))
  if takes_config_name and "config_name" not in static_argnames:
    static_argnames += ("config_name",)
  return jax.jit(func, static_argnames=static_argnames or None)


class _Block(NamedTuple):
  """Outcome of one timed block of one implementation.

  Attributes:
    outputs: Output of each sample in the block, in sample order.
    per_sample_sec: Wall time of each sample in the block.
    active_sec: Wall time of the whole block, separators included.
  """

  outputs: list[Any]
  per_sample_sec: list[float]
  active_sec: float


class _PreparedImplementation:
  """One implementation, compiled and ready to time in blocks.

  The constructor generates the warmup inputs and the inputs of the first
  block, and compiles the callable. `warm_up` runs untimed warmup samples, and
  `run_block` times a range of samples inside a trace of its own. Splitting
  the work this way lets `profile_funcs` interleave the timed blocks of
  several implementations.

  The inputs of a later block are generated when that block runs, and the
  inputs of a block are released when it is done. So at most one block of
  inputs per implementation is on device at a time, which is never more than
  when every sample ran in one block.

  Attributes:
    compilation_sec: Wall time of the first call, which compiles the callable.
    module_name: Name of the module that the callable compiles to, or "" if it
      is unknown.
    warmup_sec: Wall time of all warmup samples run so far, rewarms included.
  """

  def __init__(
      self,
      func: BenchmarkFunc,
      input_generator: InputGenerator,
      warmup_sample_keys: _Sequence[RandomKey],
      actual_sample_keys: _Sequence[RandomKey],
      warmup_sample_indices: _Sequence[Any],
      actual_sample_indices: _Sequence[Any],
      config_name: str,
      input_memory_spaces: _Sequence[str] | None,
      first_block: range,
  ):
    """Initializes the instance: generates the inputs and compiles `func`.

    Args:
      func: See `profile`.
      input_generator: See `profile`.
      warmup_sample_keys: See `profile`.
      actual_sample_keys: See `profile`.
      warmup_sample_indices: See `profile`.
      actual_sample_indices: See `profile`.
      config_name: See `profile`.
      input_memory_spaces: See `profile`.
      first_block: Positions of the samples of the first block, whose inputs are
        generated now, before compilation and warmup, as for one block.
    """
    self._input_generator = input_generator
    self._input_memory_spaces = input_memory_spaces
    self._accepts_sample_index = _accepts_kwarg(input_generator, "sample_index")
    self._actual_sample_keys = actual_sample_keys
    self._actual_sample_indices = actual_sample_indices
    # Passed to the callable only if it accepts it.
    self._config_kwargs = (
        {"config_name": config_name}
        if _accepts_kwarg(func, "config_name")
        else {}
    )
    self._jitted = _jit_unless_skipped(
        func, "config_name" in self._config_kwargs
    )
    self._run = func if self._jitted is None else self._jitted

    # Materialize the inputs up front, outside any timed region.
    self._warmup_inputs = [
        self._prepare_input(key, index)
        for key, index in zip(warmup_sample_keys, warmup_sample_indices)
    ]
    # Keyed by position in the timed-sample sequence. An entry is removed when
    # its sample has run.
    self._actual_inputs: dict[int, _SampleInputs] = {}
    self._prepare_block(first_block)

    self.compilation_sec = 0.0
    self.module_name = ""
    self.warmup_sec = 0.0
    seed_inputs = self._warmup_inputs or [
        self._actual_inputs[position] for position in first_block
    ]
    if seed_inputs:
      self._compile(*seed_inputs[0])

  def _prepare_input(
      self, random_key: RandomKey, sample_index: Any
  ) -> _SampleInputs:
    """Generates the inputs of one sample and waits until they are on device."""
    with jax.named_scope("prepare_input"):
      if self._accepts_sample_index:
        args, kwargs = self._input_generator(
            random_key, sample_index=sample_index
        )
      else:
        args, kwargs = self._input_generator(random_key)
      args = _constrain_to_memory_spaces(list(args), self._input_memory_spaces)
      jax.block_until_ready((args, kwargs))
      return args, kwargs

  def _invoke(self, args: _Sequence[Any], kwargs: dict[str, Any]) -> Any:
    """Calls the implementation on the inputs of one sample."""
    return self._run(*args, **kwargs, **self._config_kwargs)

  def _compile(self, args: _Sequence[Any], kwargs: dict[str, Any]) -> None:
    """Runs the first call, which compiles, and reads the module name.

    The compilation is timed off the first available sample so that its cost
    is reported apart from the warmup.

    Args:
      args: Positional inputs of the first available sample.
      kwargs: Keyword inputs of the first available sample.
    """
    start = time.perf_counter()
    jax.block_until_ready(self._invoke(args, kwargs))
    self.compilation_sec = time.perf_counter() - start
    # Outside every timed region. The trace reader uses the name to tell the
    # samples apart from other modules, such as the input moves that JAX runs
    # before a sharded computation.
    self.module_name = self._module_name_for(args, kwargs)

  def _module_name_for(
      self, args: _Sequence[Any], kwargs: dict[str, Any]
  ) -> str:
    """Returns the name of the module that the inputs compile to, or ""."""
    if self._jitted is None:
      # The callable manages its own jit boundary, so it may run any number
      # of modules of its own.
      return ""
    try:
      lowered = self._jitted.lower(*args, **kwargs, **self._config_kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.warning("Cannot lower the callable again.", exc_info=True)
      return ""
    return _module_name_from_lowered(lowered)

  def _prepare_block(self, positions: range) -> None:
    """Generates the inputs of the samples at `positions` that have none."""
    for position in positions:
      if position not in self._actual_inputs:
        self._actual_inputs[position] = self._prepare_input(
            self._actual_sample_keys[position],
            self._actual_sample_indices[position],
        )

  def warm_up(self, count: int | None = None) -> None:
    """Runs warmup samples untimed, and adds their wall time to `warmup_sec`.

    Args:
      count: How many of the warmup samples to run. None runs all of them.
    """
    inputs = (
        self._warmup_inputs if count is None else self._warmup_inputs[:count]
    )
    start = time.perf_counter()
    for args, kwargs in inputs:
      jax.block_until_ready(self._invoke(args, kwargs))
    self.warmup_sec += time.perf_counter() - start

  def run_block(
      self, positions: range, trace_dir: str, rewarm_samples: int = 0
  ) -> _Block:
    """Times a contiguous range of samples inside one trace.

    Args:
      positions: Positions of the samples in the timed-sample sequence. Each
        sample runs once at most.
      trace_dir: Destination of the trace for this block.
      rewarm_samples: Warmup samples to run, untimed, after the inputs of the
        block are generated and before the trace opens.

    Returns:
      The outputs and wall times of the samples, in order.
    """
    self._prepare_block(positions)
    if rewarm_samples:
      self.warm_up(rewarm_samples)
    outputs: list[Any] = []
    per_sample_sec: list[float] = []
    os.makedirs(trace_dir, exist_ok=True)
    start = time.perf_counter()
    with trace_context(trace_dir):
      for position in positions:
        args, kwargs = self._actual_inputs.pop(position)
        if position != positions.start:
          # Force a trace-event boundary between samples. Deliberately outside
          # the per-sample timer below.
          jax.block_until_ready(dummy_op())
        # The annotation names the position in the whole sequence, so a sample
        # has the same name whichever block runs it.
        with jax.profiler.TraceAnnotation(f"xloop_sample_{position}"):
          sample_start = time.perf_counter()
          result = self._invoke(args, kwargs)
          jax.block_until_ready(result)
          per_sample_sec.append(time.perf_counter() - sample_start)
        outputs.append(result)
    active_sec = time.perf_counter() - start
    return _Block(outputs, per_sample_sec, active_sec)


def profile(
    func: BenchmarkFunc,
    input_generator: InputGenerator,
    warmup_sample_keys: _Sequence[RandomKey],
    actual_sample_keys: _Sequence[RandomKey],
    warmup_sample_indices: _Sequence[Any],
    actual_sample_indices: _Sequence[Any],
    profile_subdir: str,
    config_name: str,
    input_memory_spaces: _Sequence[str] | None = None,
    output_memory_spaces: _Sequence[str] | None = None,
) -> ProfileResult:
  """Compiles, warms up and times one implementation in a single trace.

  Inputs for every sample are generated and materialized on device *before*
  the timed region opens, so the measurement covers the computation alone.

  Args:
    func: The callable under test.
    input_generator: Returns `(args, kwargs)` for a sample. May optionally
      accept a `sample_index` keyword.
    warmup_sample_keys: PRNG keys for warmup samples.
    actual_sample_keys: PRNG keys for timed samples.
    warmup_sample_indices: Sample indices paired with `warmup_sample_keys`.
    actual_sample_indices: Sample indices paired with `actual_sample_keys`.
    profile_subdir: Directory for this implementation's trace.
    config_name: Input-generation config name, passed to `func` if it accepts a
      `config_name` keyword.
    input_memory_spaces: Optional per-argument memory-space names.
    output_memory_spaces: Optional per-output memory-space names. Accepted for
      signature parity; placement of outputs is left to the compiler.

  Returns:
    The measurements and per-sample outputs.
  """
  del output_memory_spaces  # Accepted for parity; the compiler decides.
  implementation = _PreparedImplementation(
      func,
      input_generator,
      warmup_sample_keys,
      actual_sample_keys,
      warmup_sample_indices,
      actual_sample_indices,
      config_name,
      input_memory_spaces,
      first_block=range(len(actual_sample_keys)),
  )
  # Warmup. Everything is compiled by now; this settles caches and clocks.
  implementation.warm_up()
  block = implementation.run_block(
      range(len(actual_sample_keys)), profile_subdir
  )
  return ProfileResult(
      outputs=block.outputs,
      trace_dir=profile_subdir,
      compilation_sec=implementation.compilation_sec,
      warmup_sec=implementation.warmup_sec,
      active_sec=block.active_sec,
      per_sample_sec=block.per_sample_sec,
      module_name=implementation.module_name,
  )


def block_order(
    roles: _Sequence[str], num_blocks: int = DEFAULT_NUM_BLOCKS
) -> list[tuple[str, int]]:
  """Returns the order in which `profile_funcs` runs the timed blocks.

  Even passes go through the roles forward and odd passes go backward, so two
  blocks give A B B A for two roles and A B C C B A for three. A slow drift of
  the device over the run, for example of its clocks or temperature, then
  adds about the same time to every role.

  Args:
    roles: The roles, in the order they were given.
    num_blocks: Number of blocks per role.

  Returns:
    `(role, block index)` pairs. The blocks of each role appear in increasing
    order.
  """
  order: list[tuple[str, int]] = []
  for block in range(num_blocks):
    names = roles if block % 2 == 0 else list(reversed(roles))
    order.extend((name, block) for name in names)
  return order


def sample_blocks(num_samples: int, num_blocks: int) -> list[range]:
  """Splits the timed-sample positions into contiguous blocks.

  Args:
    num_samples: Number of timed samples.
    num_blocks: Requested number of blocks. There are never more blocks than
      samples, and there is always at least one block.

  Returns:
    One range of positions per block, in order. The sizes differ by 1 at most.
  """
  num_blocks = max(1, min(num_blocks, num_samples))
  bounds = [num_samples * k // num_blocks for k in range(num_blocks + 1)]
  return [range(bounds[k], bounds[k + 1]) for k in range(num_blocks)]


def block_trace_dir(directory: str, role: str, block: int) -> str:
  """Returns the trace directory of one block of one role.

  Block 0 writes to `<role>_profile`, where a run with one block also writes.
  Later blocks write to siblings, `<role>_profile_block<k>`, and not to
  subdirectories: the trace reader takes the newest trace anywhere under a
  directory, so a nested block would hide block 0.

  Args:
    directory: Parent directory for per-role trace directories.
    role: The role.
    block: The block index.

  Returns:
    The directory path.
  """
  name = f"{role}_profile" if block == 0 else f"{role}_profile_block{block}"
  return os.path.join(directory, name)


def _append_block(
    results: dict[str, Any], role: str, trace_dir: str, block: _Block
) -> None:
  """Adds the outcome of one block of `role` to the `profile_funcs` results."""
  results[f"{role}_outputs"].extend(block.outputs)
  results[f"{role}_per_sample_sec"].extend(block.per_sample_sec)
  results[f"{role}_active_sec"] += block.active_sec
  results[f"{role}_trace_dirs"].append(trace_dir)
  results[f"{role}_block_sample_counts"].append(len(block.per_sample_sec))


def profile_funcs(
    funcs: _Sequence[tuple[str, BenchmarkFunc]],
    input_generator: InputGenerator,
    warmup_sample_keys: _Sequence[RandomKey],
    actual_sample_keys: _Sequence[RandomKey],
    warmup_sample_indices: _Sequence[Any],
    actual_sample_indices: _Sequence[Any],
    config_name: str,
    directory: str,
    input_memory_spaces: _Sequence[str] | None = None,
    output_memory_spaces: _Sequence[str] | None = None,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
) -> dict[str, Any]:
  """Profiles several implementations in alternating blocks.

  The timed samples are split into `num_blocks` contiguous blocks, and the
  blocks of the roles run in the order of `block_order`, each in a trace of
  its own. If all samples of one role ran before the next role, a drift of
  the device over time would favor the role that runs later.

  Each role is compiled and warmed up just before its first block. Before each
  later block it runs up to `_REWARMUP_SAMPLES` of its warmup samples again,
  untimed: between its blocks the other roles ran, replaced the program and
  its data in the device caches, and could change the clock state, and the
  inputs of the block were generated on the device. Without the rewarm, the
  first sample of the block would pay for that. Two samples are enough
  because the program is already compiled.

  Every role sees the same sample keys in the same positions, so sample i of
  one role still pairs with sample i of another. A single role runs in one
  block, as there is nothing to alternate with.

  Args:
    funcs: `(role, callable)` pairs, e.g. `[('reference', f), ('custom', g)]`.
    input_generator: Shared input generator.
    warmup_sample_keys: PRNG keys for warmup samples.
    actual_sample_keys: PRNG keys for timed samples.
    warmup_sample_indices: Indices paired with `warmup_sample_keys`.
    actual_sample_indices: Indices paired with `actual_sample_keys`.
    config_name: Input-generation config name.
    directory: Parent directory for per-role trace subdirectories.
    input_memory_spaces: Optional per-argument memory-space names.
    output_memory_spaces: Optional per-output memory-space names.
    num_blocks: Timed blocks per role when there is more than one role. Capped
      at the number of samples.

  Returns:
    A dictionary keyed by `{role}_outputs`, `{role}_trace_dir`,
    `{role}_trace_dirs`, `{role}_block_sample_counts`, `{role}_skip_jit`,
    `{role}_compilation_sec`, `{role}_warmup_sec`, `{role}_active_sec`,
    `{role}_per_sample_sec` and `{role}_module_name`. Outputs and per-sample
    times are in sample order. `{role}_trace_dir` is the trace of block 0, and
    `{role}_trace_dirs` lists the trace of every block, with the number of
    samples of each in `{role}_block_sample_counts`. `{role}_warmup_sec`
    includes the rewarm samples, and `{role}_active_sec` is the sum over the
    blocks.

  Raises:
    ValueError: If `num_blocks` is less than 1.
  """
  if num_blocks < 1:
    raise ValueError(f"num_blocks must be at least 1, got {num_blocks}")
  del output_memory_spaces  # Accepted for parity; the compiler decides.
  roles = [name for name, _ in funcs]
  by_role = dict(funcs)
  blocks = sample_blocks(
      len(actual_sample_keys), num_blocks if len(roles) > 1 else 1
  )

  prepare = functools.partial(
      _PreparedImplementation,
      input_generator=input_generator,
      warmup_sample_keys=warmup_sample_keys,
      actual_sample_keys=actual_sample_keys,
      warmup_sample_indices=warmup_sample_indices,
      actual_sample_indices=actual_sample_indices,
      config_name=config_name,
      input_memory_spaces=input_memory_spaces,
  )
  implementations: dict[str, _PreparedImplementation] = {}
  results: dict[str, Any] = {}
  for name in roles:
    results.update({
        f"{name}_outputs": [],
        f"{name}_trace_dirs": [],
        f"{name}_block_sample_counts": [],
        f"{name}_active_sec": 0.0,
        f"{name}_per_sample_sec": [],
    })
  for name, block in block_order(roles, len(blocks)):
    implementation = implementations.get(name)
    rewarm_samples = _REWARMUP_SAMPLES
    if implementation is None:
      # Prepared just before its first block rather than all up front, so no
      # other role's work runs between the warmup and the first block.
      implementation = prepare(by_role[name], first_block=blocks[block])
      implementation.warm_up()
      implementations[name] = implementation
      rewarm_samples = 0
    trace_dir = block_trace_dir(directory, name, block)
    outcome = implementation.run_block(blocks[block], trace_dir, rewarm_samples)
    _append_block(results, name, trace_dir, outcome)

  for name in roles:
    implementation = implementations[name]
    results.update({
        f"{name}_trace_dir": results[f"{name}_trace_dirs"][0],
        f"{name}_skip_jit": bool(getattr(by_role[name], "skip_jit", False)),
        f"{name}_compilation_sec": implementation.compilation_sec,
        f"{name}_warmup_sec": implementation.warmup_sec,
        f"{name}_module_name": implementation.module_name,
    })
  return results


def check_element_equal(a: Any, b: Any, rtol: float, atol: float) -> bool:
  """Compares two output pytrees elementwise.

  Args:
    a: Expected value.
    b: Actual value.
    rtol: Relative tolerance.
    atol: Absolute tolerance.

  Returns:
    True if the trees have the same structure and all leaves are close.
  """
  try:
    a_leaves, a_def = jax.tree.flatten(a)
    b_leaves, b_def = jax.tree.flatten(b)
    if a_def != b_def:
      logger.error("Output mismatch: tree structure %s vs %s", a_def, b_def)
      return False
    for a_leaf, b_leaf in zip(a_leaves, b_leaves):
      a_np = np.asarray(jax.device_get(a_leaf))
      b_np = np.asarray(jax.device_get(b_leaf))
      if a_np.shape != b_np.shape:
        logger.error("Output mismatch: shape %s vs %s", a_np.shape, b_np.shape)
        return False
      if not np.allclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True):
        logger.error(
            "Output mismatch: max abs diff %s exceeds atol=%s rtol=%s",
            np.max(np.abs(a_np.astype(np.float64) - b_np.astype(np.float64)))
            if a_np.size
            else 0.0,
            atol,
            rtol,
        )
        return False
    return True
  except Exception:  # pylint: disable=broad-exception-caught
    logger.exception("Failed to check output equal")
    return False


def check_equal(
    reference_outputs: _Sequence[Any] | None,
    custom_outputs: _Sequence[Any] | None,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> list[bool] | bool:
  """Compares per-sample outputs of two implementations.

  Args:
    reference_outputs: Outputs from the oracle.
    custom_outputs: Outputs from the implementation under test.
    rtol: Relative tolerance.
    atol: Absolute tolerance.

  Returns:
    A per-sample list of booleans, or False if the two sequences cannot be
    compared at all.
  """
  if reference_outputs is None or custom_outputs is None:
    return False
  if len(reference_outputs) != len(custom_outputs):
    logger.error(
        "Output length mismatch: reference_outputs=%s, custom_outputs=%s",
        len(reference_outputs),
        len(custom_outputs),
    )
    return False
  if not reference_outputs or not custom_outputs:
    return False
  return [
      check_element_equal(a, b, rtol, atol)
      for a, b in zip(reference_outputs, custom_outputs)
  ]


def benchmark_func_for_model(
    model_identifier: ModelIdentifier,
    input_generator: InputGenerator,
    get_init_inputs: _Callable[[], _Sequence[Any]] | None = None,
) -> BenchmarkFunc:
  """Turns a model identifier into a callable that can be benchmarked.

  A plain function is returned as-is. A class is assumed to be a Flax linen or
  Flax NNX module and is instantiated and initialized. Flax is optional: if it
  is not installed, a class is instantiated directly from its init inputs.

  Args:
    model_identifier: The object returned by the kernel's
      `get_model_identifier()`.
    input_generator: Used to build sample arguments for Flax linen `init`.
    get_init_inputs: Returns positional arguments for the constructor.

  Returns:
    A callable ready to be profiled.
  """
  if not inspect.isclass(model_identifier):
    return model_identifier

  init_args = list(get_init_inputs()) if get_init_inputs else []

  nnx = _try_import("flax.nnx")
  if nnx is not None and issubclass(model_identifier, nnx.Module):
    try:
      rngs = nnx.Rngs({"params": jax.random.key(42)})
      return model_identifier(*init_args, rngs=rngs)
    except TypeError:
      return model_identifier(*init_args)

  model = model_identifier(*init_args)
  if hasattr(model, "init") and hasattr(model, "apply"):
    if _accepts_kwarg(input_generator, "sample_index"):
      sample_args, sample_kwargs = input_generator(
          jax.random.key(0), sample_index=0
      )
    else:
      sample_args, sample_kwargs = input_generator(jax.random.key(0))
    variables = model.init(
        {"params": jax.random.key(42)}, *sample_args, **sample_kwargs
    )
    return functools.partial(model.apply, variables)

  return cast(BenchmarkFunc, model)


def serialize_results(results: dict[str, Any], directory: str) -> None:
  """Writes `results.json` into `directory`, dropping the output arrays.

  Args:
    results: The dictionary produced by `profile_funcs`, possibly augmented with
      correctness verdicts.
    directory: Destination directory, created if absent.
  """
  serializable = {
      k: v for k, v in results.items() if not k.endswith("_outputs")
  }
  os.makedirs(directory, exist_ok=True)
  with open(os.path.join(directory, "results.json"), "w") as f:
    json.dump(serializable, f, indent=2, default=str, sort_keys=True)


def sample_keys(
    seed: int, num_warmups: int, num_samples: int
) -> tuple[list[RandomKey], list[RandomKey]]:
  """Splits a seed into warmup and timed-sample PRNG keys.

  The seed is fixed across implementations so every one of them sees
  byte-identical inputs, which is what makes the correctness comparison
  meaningful and the timings comparable.

  Args:
    seed: PRNG seed.
    num_warmups: Number of warmup samples.
    num_samples: Number of timed samples.

  Returns:
    A tuple of warmup keys and timed-sample keys.
  """
  keys = jax.random.split(jax.random.key(seed), num_warmups + num_samples)
  return list(keys[:num_warmups]), list(keys[num_warmups:])


def _package_version(*names: str) -> str:
  """Returns the installed version of the first named distribution found."""
  for name in names:
    try:
      return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      continue
    except Exception:  # pylint: disable=broad-exception-caught
      continue
  return ""


def _topology(devices: _Sequence[Any]) -> str:
  """Returns the extent of the device coordinate grid, e.g. "2x4x1"."""
  coords = [getattr(d, "coords", None) for d in devices]
  coords = [tuple(c) for c in coords if c is not None]
  if not coords or len({len(c) for c in coords}) != 1:
    return ""
  extents = [
      max(c[axis] for c in coords) - min(c[axis] for c in coords) + 1
      for axis in range(len(coords[0]))
  ]
  return "x".join(str(e) for e in extents)


def _flags_from_env() -> list[str]:
  """Returns the compiler flags set through the environment, in order."""
  flags: list[str] = []
  for var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
    raw = os.environ.get(var, "")
    try:
      flags.extend(shlex.split(raw))
    except ValueError:
      flags.extend(raw.split())
  return flags


def describe_environment() -> dict[str, Any]:
  """Records the software and hardware a measurement is taken on.

  Two results are only comparable if they were measured on the same kind of
  device with the same software and flags, so the engine stores this record
  beside every result. Each entry is best effort: anything that cannot be
  determined is left empty rather than failing the run.

  Returns:
    A dictionary keyed by the field names of the `RunEnvironment` message.
  """
  env: dict[str, Any] = {
      "python_version": platform.python_version(),
      "jax_version": _package_version("jax"),
      "jaxlib_version": _package_version("jaxlib"),
      "libtpu_version": _package_version("libtpu", "libtpu-nightly"),
      "xprof_version": _package_version("xprof", "xprof-nightly"),
      "accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE", ""),
      "xla_flags": _flags_from_env(),
  }
  try:
    env["host_name"] = socket.gethostname()
  except OSError:
    env["host_name"] = ""
  if not env["jax_version"]:
    env["jax_version"] = getattr(jax, "__version__", "")
  try:
    devices = jax.devices()
  except Exception:  # pylint: disable=broad-exception-caught
    logger.exception("Could not list devices")
    devices = []
  if devices:
    first = devices[0]
    env["platform"] = str(getattr(first, "platform", ""))
    env["device_kind"] = str(getattr(first, "device_kind", ""))
    env["device_count"] = len(devices)
    env["topology"] = _topology(devices)
    client = getattr(first, "client", None)
    env["platform_version"] = str(getattr(client, "platform_version", ""))
  return env
