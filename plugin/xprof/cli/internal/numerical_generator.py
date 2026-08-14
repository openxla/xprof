"""Reusable library for generating dtype-aware heavy-tailed test tensors."""

import collections.abc
import dataclasses
import importlib
import io
import json
import logging
import os
from typing import Any, BinaryIO
import ml_dtypes
import numpy as np

_Sequence = collections.abc.Sequence
_Iterable = collections.abc.Iterable


@dataclasses.dataclass(frozen=True)
class DtypeProfile:
  dtype_str: str
  numpy_dtype: Any
  mantissa_bits: int
  exponent_bits: int
  max_finite: float
  min_normal: float
  min_subnormal: float
  default_df: float  # Student's t degrees of freedom (2.5 - 4.0)
  outlier_scale_limit: float


PROFILES: dict[str, DtypeProfile] = {
    "float32": DtypeProfile(
        dtype_str="float32",
        numpy_dtype=np.float32,
        mantissa_bits=23,
        exponent_bits=8,
        max_finite=3.4028235e38,
        min_normal=1.17549435e-38,
        min_subnormal=1.4012985e-45,
        default_df=4.0,
        outlier_scale_limit=1e6,
    ),
    "bfloat16": DtypeProfile(
        dtype_str="bfloat16",
        numpy_dtype=ml_dtypes.bfloat16,
        mantissa_bits=7,
        exponent_bits=8,
        max_finite=3.3895314e38,
        min_normal=1.17549435e-38,
        min_subnormal=9.18355e-41,
        default_df=3.0,
        outlier_scale_limit=100.0,
    ),
    "float16": DtypeProfile(
        dtype_str="float16",
        numpy_dtype=np.float16,
        mantissa_bits=10,
        exponent_bits=5,
        max_finite=65504.0,
        min_normal=6.1035156e-5,
        min_subnormal=5.9604645e-8,
        default_df=3.5,
        outlier_scale_limit=20.0,
    ),
    "fp8_e4m3": DtypeProfile(
        dtype_str="fp8_e4m3",
        numpy_dtype=ml_dtypes.float8_e4m3fn,
        mantissa_bits=3,
        exponent_bits=4,
        max_finite=448.0,
        min_normal=0.015625,
        min_subnormal=0.001953125,
        default_df=2.5,
        outlier_scale_limit=5.0,
    ),
    "fp8_e5m2": DtypeProfile(
        dtype_str="fp8_e5m2",
        numpy_dtype=ml_dtypes.float8_e5m2,
        mantissa_bits=2,
        exponent_bits=5,
        max_finite=57344.0,
        min_normal=6.1035156e-5,
        min_subnormal=1.5258789e-5,
        default_df=3.0,
        outlier_scale_limit=50.0,
    ),
}


def generate_student_t_tensor(
    shape: _Sequence[int],
    dtype_str: str = "bfloat16",
    df: float | None = None,
    seed: int = 42,
) -> np.ndarray:
  """Generates a heavy-tailed tensor from Student's t-distribution."""
  if df is not None and (not np.isfinite(df) or df <= 0.0):
    raise ValueError(
        f"degrees_of_freedom must be finite and positive, got {df}"
    )
  if dtype_str not in PROFILES:
    raise KeyError(
        f"Unsupported dtype_str '{dtype_str}'. Supported:"
        f" {list(PROFILES.keys())}"
    )
  profile = PROFILES[dtype_str]
  degrees_of_freedom = df if df is not None else profile.default_df
  rng = np.random.default_rng(seed)

  z = rng.standard_normal(shape)
  v = rng.gamma(degrees_of_freedom / 2.0, scale=2.0, size=shape)
  v = np.maximum(v, 1e-10)
  t_dist = z / np.sqrt(v / degrees_of_freedom)
  # Prevent extreme tail draws from overflowing to NaN (fp8_e4m3) or Inf (f16).
  max_bound = float(profile.max_finite * 0.95)
  t_clipped = np.clip(t_dist, -max_bound, max_bound)
  return t_clipped.astype(profile.numpy_dtype)


def generate_outlier_tensor(
    shape: _Sequence[int],
    dtype_str: str = "bfloat16",
    outlier_ratio: float = 0.01,
    outlier_scale: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
  """Generates a tensor with heavy-tailed localized activation spikes."""
  if not (0.0 <= outlier_ratio <= 1.0):
    raise ValueError(
        f"outlier_ratio must be between 0.0 and 1.0, got {outlier_ratio}"
    )
  if not np.isfinite(outlier_scale) or outlier_scale <= 0.0:
    raise ValueError(
        f"outlier_scale must be finite and positive, got {outlier_scale}"
    )
  if dtype_str not in PROFILES:
    raise KeyError(
        f"Unsupported dtype_str '{dtype_str}'. Supported:"
        f" {list(PROFILES.keys())}"
    )
  profile = PROFILES[dtype_str]
  mask_rng = np.random.default_rng(seed + 10007)

  base = generate_student_t_tensor(shape, dtype_str, seed=seed)
  base_fp32 = base.astype(np.float32)
  mask = mask_rng.uniform(size=shape) < outlier_ratio
  signs = mask_rng.choice(np.array([-1.0, 1.0]), size=shape)

  scale = min(outlier_scale, profile.outlier_scale_limit)
  median_val = np.median(np.abs(base_fp32)) + 1e-6
  outliers = signs * (median_val * scale)

  return np.where(mask, outliers, base_fp32).astype(profile.numpy_dtype)


def generate_cancellation_tensor(
    shape: _Sequence[int],
    dtype_str: str = "bfloat16",
    reduction_axis: int = -1,
    epsilon: float = 0.1,
) -> np.ndarray:
  """Generates structured cancellation pairs scaled safely for dtype."""
  if reduction_axis < -len(shape) or reduction_axis >= len(shape):
    raise IndexError(
        f"reduction_axis {reduction_axis} out of bounds for shape {shape}"
    )
  dim_len = shape[reduction_axis]
  if dim_len % 2 != 0:
    raise ValueError(
        f"reduction_axis {reduction_axis} dimension length {dim_len} must be"
        " even for alternating cancellation pairs."
    )
  if dtype_str not in PROFILES:
    raise KeyError(
        f"Unsupported dtype_str '{dtype_str}'. Supported:"
        f" {list(PROFILES.keys())}"
    )
  profile = PROFILES[dtype_str]
  large_val = min(1000.0, float(profile.max_finite * 0.1))

  # Ensure epsilon is at least 2 ULPs of large_val in target precision
  # so that quantization does not obliterate the cancellation residual.
  if dtype_str == "float32":
    ulp_step = float(np.spacing(np.float32(large_val)))
  else:
    exp_val = np.frexp(large_val)[1]
    ulp_step = float(2.0 ** (exp_val - profile.mantissa_bits))
  effective_eps = max(epsilon, ulp_step * 2.0)

  arr = np.zeros(shape, dtype=np.float32)

  slices_even = [slice(None)] * len(shape)
  slices_odd = [slice(None)] * len(shape)
  slices_even[reduction_axis] = slice(0, None, 2)
  slices_odd[reduction_axis] = slice(1, None, 2)

  arr[tuple(slices_even)] = large_val
  arr[tuple(slices_odd)] = -large_val + effective_eps
  return arr.astype(profile.numpy_dtype)


def generate_boundary_probe_tensor(
    shape: _Sequence[int],
    dtype_str: str = "bfloat16",
    tile_stride: int = 128,
) -> np.ndarray:
  """Generates boundary probes distributed across TPU VMEM tile strides."""
  if dtype_str not in PROFILES:
    raise KeyError(
        f"Unsupported dtype_str '{dtype_str}'. Supported:"
        f" {list(PROFILES.keys())}"
    )
  profile = PROFILES[dtype_str]
  if dtype_str == "float32":
    probes = [1e4, -1e4, profile.min_normal, profile.min_subnormal, 0.0]
  elif dtype_str == "bfloat16":
    probes = [1e3, -1e3, profile.min_normal, profile.min_subnormal, 0.0]
  elif dtype_str == "float16":
    probes = [1000.0, -1000.0, profile.min_normal, profile.min_subnormal, 0.0]
  elif dtype_str == "fp8_e4m3":
    probes = [100.0, -100.0, profile.min_normal, profile.min_subnormal, 0.0]
  elif dtype_str == "fp8_e5m2":
    probes = [1000.0, -1000.0, profile.min_normal, profile.min_subnormal, 0.0]
  else:
    probes = [1e4, -1e4, profile.min_normal, profile.min_subnormal, 0.0]

  total_size = int(np.prod(shape))
  arr = np.zeros(total_size, dtype=np.float32)
  for idx, p in enumerate(probes):
    arr[idx :: (len(probes) * tile_stride)] = p

  return arr.reshape(shape).astype(profile.numpy_dtype)


def _resolve_dtype(dtype_str: str) -> np.dtype:
  """Resolves string representation to numpy dtype, including ml_dtypes."""
  if dtype_str in ("bfloat16", "ml_dtypes.bfloat16"):
    return np.dtype(ml_dtypes.bfloat16)
  if dtype_str in ("float8_e4m3fn", "fp8_e4m3", "ml_dtypes.float8_e4m3fn"):
    return np.dtype(ml_dtypes.float8_e4m3fn)
  if dtype_str in ("float8_e5m2", "fp8_e5m2", "ml_dtypes.float8_e5m2"):
    return np.dtype(ml_dtypes.float8_e5m2)
  return np.dtype(dtype_str)


def save_test_suite(
    suite: list[dict[str, Any]],
    target: str | os.PathLike[str] | BinaryIO,
) -> None:
  """Serializes a test suite (args, kwargs, metadata) to a compressed archive.

  Args:
    suite: List of test batch dicts with 'name', 'args', 'kwargs', and 'regime'.
    target: Destination file path (str/PathLike) or binary stream (BinaryIO).
  """
  metadata = {
      "version": 2,
      "num_batches": len(suite),
      "batches": [],
  }
  arrays_to_save: dict[str, Any] = {}

  for i, item in enumerate(suite):
    args_list = item.get("args", ())
    kwargs_dict = item.get("kwargs", {})

    arg_metadata = []
    for j, arg in enumerate(args_list):
      arr = np.ascontiguousarray(arg)
      arg_metadata.append({
          "shape": list(arr.shape),
          "dtype": str(arr.dtype),
      })
      arrays_to_save[f"batch_{i}__arg_{j}"] = arr.view(np.uint8)

    tensor_kwargs_meta = {}
    scalar_kwargs_meta = {}
    for k, v in kwargs_dict.items():
      if isinstance(v, np.ndarray) or hasattr(v, "__array__"):
        arr = np.ascontiguousarray(v)
        tensor_kwargs_meta[k] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        arrays_to_save[f"batch_{i}__kwarg_{k}"] = arr.view(np.uint8)
      else:
        scalar_kwargs_meta[k] = v

    b_meta = {
        "index": i,
        "name": item.get("name", f"batch_{i}"),
        "regime": item.get("regime", "unknown"),
        "args": arg_metadata,
        "tensor_kwargs": tensor_kwargs_meta,
        "scalar_kwargs": scalar_kwargs_meta,
    }
    metadata["batches"].append(b_meta)

  arrays_to_save["__metadata__"] = np.array(json.dumps(metadata))

  if isinstance(target, (str, os.PathLike)):
    path_str = os.fspath(target)
    dir_path = os.path.dirname(os.path.abspath(path_str))
    if dir_path:
      os.makedirs(dir_path, exist_ok=True)
    with open(path_str, "wb") as f:
      np.savez_compressed(f, **arrays_to_save)
  else:
    np.savez_compressed(target, **arrays_to_save)


def load_test_suite(
    source: str | os.PathLike[str] | BinaryIO | bytes,
    as_jax_arrays: bool = False,
) -> list[dict[str, Any]]:
  """Loads and reconstructs a test suite from a .npz archive, resource, or stream.

  Args:
    source: Filepath (str/PathLike), Google3 resource path, stream, or bytes.
    as_jax_arrays: If True, converts loaded NumPy arrays to JAX device arrays.

  Returns:
    Reconstructed test suite list of test batch dicts.
  """
  if isinstance(source, bytes):
    data = np.load(io.BytesIO(source))
  elif isinstance(source, (str, os.PathLike)):
    path_str = os.fspath(source)
    if os.path.exists(path_str):
      data = np.load(path_str)
    else:
      # Fallback to dynamic hermetic test resource lookup if available
      try:
        pyglib_resources = importlib.import_module("google3.pyglib.resources")
        raw_bytes = pyglib_resources.GetResource(path_str, mode="rb")
        data = np.load(io.BytesIO(raw_bytes))
      except (ImportError, OSError, IOError, ValueError, KeyError):
        data = np.load(path_str)
  else:
    data = np.load(source)

  try:
    meta_raw = data["__metadata__"].item()
    metadata = json.loads(meta_raw)

    suite = []
    for b_info in metadata["batches"]:
      i = b_info["index"]
      raw_args = []
      for j, a_meta in enumerate(b_info["args"]):
        raw_u8 = data[f"batch_{i}__arg_{j}"]
        dt = _resolve_dtype(a_meta["dtype"])
        raw_args.append(raw_u8.view(dt).reshape(a_meta["shape"]))

      raw_kwargs = {}
      for k, kw_meta in b_info.get("tensor_kwargs", {}).items():
        raw_u8 = data[f"batch_{i}__kwarg_{k}"]
        dt = _resolve_dtype(kw_meta["dtype"])
        raw_kwargs[k] = raw_u8.view(dt).reshape(kw_meta["shape"])

      raw_kwargs.update(b_info.get("scalar_kwargs", {}))

      if as_jax_arrays:
        jnp = importlib.import_module("jax.numpy")
        args = tuple(jnp.asarray(a) for a in raw_args)
        kwargs = {
            k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
            for k, v in raw_kwargs.items()
        }
      else:
        args = tuple(raw_args)
        kwargs = raw_kwargs

      suite.append({
          "name": b_info["name"],
          "args": args,
          "kwargs": kwargs,
          "regime": b_info["regime"],
      })
    return suite
  finally:
    if hasattr(data, "close"):
      data.close()


def _generate_procedural_suite(
    shapes: _Sequence[_Sequence[int]],
    dtype_str: str = "bfloat16",
    tier: str = "presubmit",
    seed: int = 42,
    as_jax_arrays: bool = False,
) -> list[dict[str, Any]]:
  """Generates procedural test suite using statistical distributions."""
  if tier == "fast_agent":
    num_student_t, num_outliers = 2, 1
  elif tier == "deep_fuzzing":
    num_student_t, num_outliers = 30, 15
  else:  # presubmit default
    num_student_t, num_outliers = 6, 3

  def _convert(arr: np.ndarray) -> Any:
    if as_jax_arrays:
      jnp = importlib.import_module("jax.numpy")
      return jnp.asarray(arr)
    return arr

  suite = []
  for b in range(num_student_t):
    tensors = [
        _convert(
            generate_student_t_tensor(s, dtype_str, seed=seed + b * 10 + i)
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": f"student_t_batch_{b}",
        "args": tuple(tensors),
        "kwargs": {},
        "regime": "student_t",
    })

  for b in range(num_outliers):
    scale = 20.0 + b * 30.0
    tensors = [
        _convert(
            generate_outlier_tensor(
                s,
                dtype_str,
                outlier_scale=scale,
                seed=seed + 100 + b * 10 + i,
            )
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": f"outliers_scale_{int(scale)}x_batch_{b}",
        "args": tuple(tensors),
        "kwargs": {},
        "regime": "outliers",
    })

  cancellation_tensors = []
  for s in shapes:
    if not s or s[-1] % 2 != 0:
      cancellation_tensors.append(
          _convert(
              generate_student_t_tensor(s, dtype_str, df=2.5, seed=seed + 999)
          )
      )
    else:
      cancellation_tensors.append(
          _convert(generate_cancellation_tensor(s, dtype_str))
      )
  suite.append({
      "name": "cancellation_pairs",
      "args": tuple(cancellation_tensors),
      "kwargs": {},
      "regime": "cancellation",
  })

  boundary_tensors = [
      _convert(generate_boundary_probe_tensor(s, dtype_str)) for s in shapes
  ]
  suite.append({
      "name": "boundary_probes",
      "args": tuple(boundary_tensors),
      "kwargs": {},
      "regime": "boundary",
  })

  return suite


def generate_test_suite(
    shapes: _Sequence[int] | _Sequence[_Sequence[int]],
    dtype_str: str = "bfloat16",
    tier: str = "presubmit",
    seed: int = 42,
    persisted_path: str | os.PathLike[str] | None = None,
    mode: str = "auto",
    as_jax_arrays: bool = False,
) -> list[dict[str, Any]]:
  """Generates multi-regime test suite with optional persistence support.

  By default (persisted_path=None), performs 100% in-memory procedural
  generation with zero disk I/O.

  Args:
    shapes: Single shape tuple or sequence of shape tuples.
    dtype_str: Data type string ("bfloat16", "float32", "float16", "fp8_e4m3",
      "fp8_e5m2").
    tier: Test thoroughness tier ("fast_agent", "presubmit", "deep_fuzzing").
    seed: Random number generator seed.
    persisted_path: Optional path to a .npz file or resource to load/save.
    mode: Persistence mode ("auto", "read_only", "record").
    as_jax_arrays: If True, returns arrays as jax.Array (jnp.ndarray).

  Returns:
    List of test batch dicts with 'name', 'args', 'kwargs', and 'regime'.

  Raises:
    RuntimeError: If mode is 'read_only' and persisted_path does not exist.
  """
  shape_list: list[_Sequence[int]] = []
  shapes_any: Any = shapes
  if isinstance(shapes, tuple) and not shapes:
    shape_list.append(())
  elif shapes and isinstance(shapes[0], (int, np.integer)):
    shape_list.append(tuple(int(x) for x in shapes_any))
  else:
    for s in shapes_any:
      shape_list.append(tuple(int(x) for x in s))

  # 1. Default path: If no persisted_path is specified, generate procedurally.
  if not persisted_path:
    return _generate_procedural_suite(
        shape_list,
        dtype_str=dtype_str,
        tier=tier,
        seed=seed,
        as_jax_arrays=as_jax_arrays,
    )

  # 2. Opt-in path: Load if fixture exists
  path_str = os.fspath(persisted_path)
  if mode == "read_only":
    if not os.path.exists(path_str) and not path_str.startswith("google3/"):
      raise RuntimeError(f"Required golden fixture does not exist: {path_str}")
    return load_test_suite(persisted_path, as_jax_arrays=as_jax_arrays)

  if mode == "auto" and (
      os.path.exists(path_str) or path_str.startswith("google3/")
  ):
    try:
      return load_test_suite(persisted_path, as_jax_arrays=as_jax_arrays)
    except (OSError, ValueError, KeyError) as e:
      logging.warning(
          "Failed to load persisted tensors from %s (%s). Regenerating.",
          path_str,
          e,
      )

  # 3. Opt-in path: Procedurally generate and optionally record
  suite = _generate_procedural_suite(
      shape_list,
      dtype_str=dtype_str,
      tier=tier,
      seed=seed,
      as_jax_arrays=as_jax_arrays,
  )

  if mode in ("auto", "record") and not os.path.exists(path_str):
    try:
      save_test_suite(suite, persisted_path)
    except (OSError, ValueError, KeyError) as e:
      logging.warning("Failed to save generated tensors to %s: %s", path_str, e)

  return suite
