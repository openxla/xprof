"""JAX/Pallas measurement harness for the XLoop Engine.

Compiles a candidate callable, runs untimed warmup iterations, executes timed
samples inside a `jax.profiler.trace` region with a separator op between
samples so executions can be matched on the device timeline, and compares
per-sample outputs against the reference implementation.

Requires `jax`. The engine imports this module lazily so that `xprof` itself
keeps working without a JAX install.
"""

from __future__ import annotations

import collections.abc
import functools
import importlib
import inspect
import json
import logging
import os
import time
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

_Callable = collections.abc.Callable
_Sequence = collections.abc.Sequence

BenchmarkFunc = _Callable[..., Any]
InputGenerator = _Callable[..., tuple[_Sequence[Any], dict[str, Any]]]
ModelIdentifier = Any
RandomKey = Any

# Comparison tolerances used when the caller supplies none.
_DEFAULT_RTOL = 1e-5
_DEFAULT_ATOL = 1e-5

logger = logging.getLogger(__name__)


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
    session_id: Optional profiling-service session identifier (None when traces
      are written to local disk).
    isolated_trace_dir: Optional secondary trace directory (empty when inputs
      are generated outside the timed region).
    isolated_session_id: Optional secondary session identifier.
    compilation_sec: Wall time of the first (compiling) call.
    warmup_sec: Wall time of all warmup samples.
    active_sec: Wall time of all timed samples.
    per_sample_sec: Wall time of each timed sample individually. The summarizer
      needs the distribution, not just the total: it sizes its
      faster/slower/no-change threshold from the spread, so an aggregate alone
      would collapse that threshold to its floor.
  """

  outputs: list[Any]
  trace_dir: str
  session_id: str | None
  isolated_trace_dir: str
  isolated_session_id: str | None
  compilation_sec: float
  warmup_sec: float
  active_sec: float
  per_sample_sec: list[float]


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
  """Compiles, warms up and times one implementation.

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

  static_argnames = tuple(get_static_argnames(func))
  accepts_config_name = _accepts_kwarg(func, "config_name")
  accepts_sample_index = _accepts_kwarg(input_generator, "sample_index")
  skip_jit = bool(getattr(func, "skip_jit", False))

  if accepts_config_name and "config_name" not in static_argnames:
    # It is a string, so it cannot be traced; it has to be a compile-time
    # constant. Kernels should not have to declare this themselves.
    static_argnames += ("config_name",)

  def call_input_generator(random_key: RandomKey, sample_index: Any) -> Any:
    if accepts_sample_index:
      return input_generator(random_key, sample_index=sample_index)
    return input_generator(random_key)

  def prepare_input(random_key: RandomKey, sample_index: Any) -> Any:
    with jax.named_scope("prepare_input"):
      args, kwargs = call_input_generator(random_key, sample_index)
      args = _constrain_to_memory_spaces(list(args), input_memory_spaces)
      jax.block_until_ready((args, kwargs))
      return args, kwargs

  if skip_jit:
    # The kernel manages its own jit boundary.
    run = func
  else:
    run = jax.jit(func, static_argnames=static_argnames or None)

  def invoke(args: _Sequence[Any], kwargs: dict[str, Any]) -> Any:
    if accepts_config_name:
      return run(*args, **kwargs, config_name=config_name)
    return run(*args, **kwargs)

  # Materialize every sample's inputs up front, outside any timed region.
  warmup_inputs = [
      prepare_input(k, i)
      for k, i in zip(warmup_sample_keys, warmup_sample_indices)
  ]
  actual_inputs = [
      prepare_input(k, i)
      for k, i in zip(actual_sample_keys, actual_sample_indices)
  ]

  # Compilation. Timed off the first available sample so the cost is
  # attributed separately rather than polluting the first warmup.
  seed_inputs = warmup_inputs or actual_inputs
  compilation_sec = 0.0
  if seed_inputs:
    args, kwargs = seed_inputs[0]
    start = time.perf_counter()
    jax.block_until_ready(invoke(args, kwargs))
    compilation_sec = time.perf_counter() - start

  # Warmup. Everything is compiled by now; this settles caches and clocks.
  start = time.perf_counter()
  for args, kwargs in warmup_inputs:
    jax.block_until_ready(invoke(args, kwargs))
  warmup_sec = time.perf_counter() - start

  outputs: list[Any] = []
  per_sample_sec: list[float] = []
  os.makedirs(profile_subdir, exist_ok=True)
  start = time.perf_counter()
  with trace_context(profile_subdir):
    for index, (args, kwargs) in enumerate(actual_inputs):
      if index:
        # Force a trace-event boundary between samples. Deliberately outside
        # the per-sample timer below.
        jax.block_until_ready(dummy_op())
      with jax.profiler.TraceAnnotation(f"xloop_sample_{index}"):
        sample_start = time.perf_counter()
        result = invoke(args, kwargs)
        jax.block_until_ready(result)
        per_sample_sec.append(time.perf_counter() - sample_start)
      outputs.append(result)
  active_sec = time.perf_counter() - start

  return ProfileResult(
      outputs=outputs,
      trace_dir=profile_subdir,
      session_id=None,
      isolated_trace_dir="",
      isolated_session_id=None,
      compilation_sec=compilation_sec,
      warmup_sec=warmup_sec,
      active_sec=active_sec,
      per_sample_sec=per_sample_sec,
  )


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
) -> dict[str, Any]:
  """Profiles several implementations and flattens the results.

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

  Returns:
    A dictionary keyed by `{role}_outputs`, `{role}_trace_dir`,
    `{role}_xprof_session_id`, `{role}_isolated_trace_dir`,
    `{role}_isolated_xprof_session_id`, `{role}_skip_jit`,
    `{role}_compilation_sec`, `{role}_warmup_sec`, `{role}_active_sec` and
    `{role}_per_sample_sec`.
  """
  results: dict[str, Any] = {}
  for name, func in funcs:
    profile_result = profile(
        func,
        input_generator,
        warmup_sample_keys,
        actual_sample_keys,
        warmup_sample_indices,
        actual_sample_indices,
        os.path.join(directory, f"{name}_profile"),
        config_name=config_name,
        input_memory_spaces=input_memory_spaces,
        output_memory_spaces=output_memory_spaces,
    )
    results.update({
        f"{name}_outputs": profile_result.outputs,
        f"{name}_trace_dir": profile_result.trace_dir,
        f"{name}_xprof_session_id": profile_result.session_id,
        f"{name}_isolated_trace_dir": profile_result.isolated_trace_dir,
        f"{name}_isolated_xprof_session_id": profile_result.isolated_session_id,
        f"{name}_skip_jit": bool(getattr(func, "skip_jit", False)),
        f"{name}_compilation_sec": profile_result.compilation_sec,
        f"{name}_warmup_sec": profile_result.warmup_sec,
        f"{name}_active_sec": profile_result.active_sec,
        f"{name}_per_sample_sec": profile_result.per_sample_sec,
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
