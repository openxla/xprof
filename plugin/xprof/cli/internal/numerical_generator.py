"""Reusable library for generating dtype-aware heavy-tailed test tensors."""

import collections.abc
import dataclasses
import importlib
import io
import json
import logging
import os
import types
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
    "float64": DtypeProfile(
        dtype_str="float64",
        numpy_dtype=np.float64,
        mantissa_bits=52,
        exponent_bits=11,
        max_finite=float(np.finfo(np.float64).max),
        min_normal=float(np.finfo(np.float64).tiny),
        min_subnormal=float(np.nextafter(np.float64(0.0), np.float64(1.0))),
        default_df=5.0,
        outlier_scale_limit=100.0,
    ),
}


@dataclasses.dataclass(frozen=True)
class IntegerDtypeProfile:
  dtype_str: str
  numpy_dtype: Any
  min_val: int
  max_val: int
  is_signed: bool


INTEGER_PROFILES: types.MappingProxyType[str, IntegerDtypeProfile] = (
    types.MappingProxyType({
        "int32": IntegerDtypeProfile(
            "int32", np.int32, -2147483648, 2147483647, True
        ),
        "int64": IntegerDtypeProfile(
            "int64",
            np.int64,
            -9223372036854775808,
            9223372036854775807,
            True,
        ),
        "int16": IntegerDtypeProfile("int16", np.int16, -32768, 32767, True),
        "int8": IntegerDtypeProfile("int8", np.int8, -128, 127, True),
        "uint32": (
            IntegerDtypeProfile("uint32", np.uint32, 0, 4294967295, False)
        ),
        "uint64": IntegerDtypeProfile(
            "uint64", np.uint64, 0, 18446744073709551615, False
        ),
        "uint16": IntegerDtypeProfile("uint16", np.uint16, 0, 65535, False),
        "uint8": IntegerDtypeProfile("uint8", np.uint8, 0, 255, False),
    })
)

INTEGER_DTYPES: frozenset[str] = frozenset(INTEGER_PROFILES.keys())
BOOLEAN_DTYPES: frozenset[str] = frozenset({"bool"})
ALL_SUPPORTED_DTYPES: frozenset[str] = frozenset(
    set(PROFILES.keys()) | INTEGER_DTYPES | BOOLEAN_DTYPES
)


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


def generate_index_tensor(
    shape: _Sequence[int],
    upper_bound: int,
    lower_bound: int = 0,
    dtype_str: str = "int32",
    include_boundaries: bool = True,
    seed: int = 42,
) -> np.ndarray:
  """Generates bounded discrete indices in [lower_bound, upper_bound - 1].

  Guarantees valid range for MoE expert IDs, embedding lookups, gather/scatter.
  When include_boundaries=True, injects lower_bound and upper_bound - 1 to test
  edge index conditions.

  Args:
    shape: Target tensor shape tuple.
    upper_bound: Exclusive upper bound for generated indices.
    lower_bound: Inclusive lower bound for generated indices (default: 0).
    dtype_str: Target integer data type string ("int32", "int64", etc.).
    include_boundaries: If True, injects lower_bound and upper_bound - 1.
    seed: Random number generator seed.

  Returns:
    NumPy array with discrete indices strictly bounded in [lower_bound,
    upper_bound - 1].

  Raises:
    ValueError: If upper_bound <= lower_bound.
  """
  if upper_bound <= lower_bound:
    raise ValueError(
        f"upper_bound ({upper_bound}) must be greater than lower_bound"
        f" ({lower_bound})"
    )
  dtype = np.dtype(dtype_str)
  rng = np.random.default_rng(seed)
  indices = rng.integers(lower_bound, upper_bound, size=shape, dtype=dtype)
  if include_boundaries and indices.size >= 1:
    flat = indices.reshape(-1)
    flat[0] = lower_bound
    if indices.size >= 2:
      flat[-1] = upper_bound - 1
    indices = flat.reshape(shape)
  return indices


def generate_segment_ids_tensor(
    shape: _Sequence[int],
    num_segments: int,
    is_sorted: bool = True,
    dtype_str: str = "int32",
    seed: int = 42,
) -> np.ndarray:
  """Generates segment IDs in [0, num_segments - 1] for segmented reductions.

  If is_sorted=True, values along the last axis are monotonically
  non-decreasing.

  Args:
    shape: Target tensor shape tuple.
    num_segments: Total number of distinct segment IDs.
    is_sorted: If True, guarantees monotonically non-decreasing segment IDs.
    dtype_str: Target integer data type string ("int32", "int64", etc.).
    seed: Random number generator seed.

  Returns:
    NumPy array with segment IDs spanning [0, num_segments - 1].

  Raises:
    ValueError: If num_segments <= 0.
  """
  if num_segments <= 0:
    raise ValueError(f"num_segments must be positive, got {num_segments}")
  dtype = np.dtype(dtype_str)
  rng = np.random.default_rng(seed)
  if not shape:
    return np.array(0, dtype=dtype)
  if num_segments == 1:
    return np.zeros(shape, dtype=dtype)
  if is_sorted:
    last_dim = shape[-1]
    if last_dim >= num_segments:
      cuts = np.sort(
          rng.choice(last_dim - 1, size=num_segments - 1, replace=False) + 1
      )
      cuts = np.concatenate(([0], cuts, [last_dim]))
      seg_1d = np.empty(last_dim, dtype=dtype)
      for seg_id in range(num_segments):
        seg_1d[cuts[seg_id] : cuts[seg_id + 1]] = seg_id
      leading_shape = shape[:-1]
      if leading_shape:
        return np.broadcast_to(seg_1d, shape).copy()
      return seg_1d
    else:
      return np.sort(
          rng.integers(0, num_segments, size=shape, dtype=dtype), axis=-1
      )
  else:
    return rng.integers(0, num_segments, size=shape, dtype=dtype)


def generate_mask_tensor(
    shape: _Sequence[int],
    mask_type: str = "causal",
    density: float = 0.5,
    seq_lens: _Sequence[int] | None = None,
    dtype_str: str = "bool",
    seed: int = 42,
) -> np.ndarray:
  """Generates boolean or binary masks: causal, bernoulli sparse, or padding.

  Args:
    shape: Target mask shape tuple.
    mask_type: Topology of mask ("causal", "bernoulli", "padding").
    density: Active ratio for Bernoulli sparse mask (default: 0.5).
    seq_lens: Optional explicit sequence lengths for padding mask.
    dtype_str: Target data type string ("bool", "int32", etc.).
    seed: Random number generator seed.

  Returns:
    NumPy array containing the structured boolean or integer mask.

  Raises:
    ValueError: If density is not in [0, 1], if causal mask shape has fewer
      than 2 dimensions, or if mask_type is unknown.
  """
  if not (0.0 <= density <= 1.0):
    raise ValueError(f"density must be in [0, 1], got {density}")
  dtype = np.dtype(dtype_str)
  rng = np.random.default_rng(seed)
  if mask_type == "causal":
    if len(shape) < 2:
      raise ValueError(
          f"Causal mask requires at least 2 dimensions, got shape {shape}"
      )
    n, m = shape[-2], shape[-1]
    causal_2d = np.tril(np.ones((n, m), dtype=dtype))
    if len(shape) > 2:
      leading_ones = (1,) * (len(shape) - 2)
      return np.broadcast_to(
          causal_2d.reshape(leading_ones + (n, m)), shape
      ).copy()
    return causal_2d
  elif mask_type == "bernoulli":
    raw_bool = rng.uniform(size=shape) < density
    return raw_bool.astype(dtype)
  elif mask_type == "padding":
    if len(shape) < 2:
      seq_len = shape[-1] if shape else 1
      lengths = seq_lens if seq_lens is not None else [max(1, seq_len // 2)]
      mask = np.arange(seq_len) < lengths[0]
      return mask.astype(dtype).reshape(shape)
    batch_size, max_len = shape[0], shape[-1]
    if seq_lens is not None:
      lengths = np.asarray(seq_lens, dtype=np.int32)
    else:
      lengths = rng.integers(1, max_len + 1, size=batch_size, dtype=np.int32)
    positions = np.arange(max_len)
    mask_2d = positions[None, :] < lengths[:, None]
    if len(shape) > 2:
      leading_ones = (1,) * (len(shape) - 2)
      return (
          np.broadcast_to(
              mask_2d.reshape((batch_size,) + leading_ones + (max_len,)), shape
          )
          .astype(dtype)
          .copy()
      )
    return mask_2d.astype(dtype)
  else:
    raise ValueError(
        f"Unknown mask_type '{mask_type}'. Supported: 'causal', 'bernoulli',"
        " 'padding'"
    )


def generate_integer_tensor(
    shape: _Sequence[int],
    dtype_str: str = "int32",
    min_val: int | None = None,
    max_val: int | None = None,
    seed: int = 42,
) -> np.ndarray:
  """Generates integer test tensors covering uniform, boundary extremes, and strides.

  Args:
    shape: Target tensor shape tuple.
    dtype_str: Target integer data type string ("int32", "int64", etc.).
    min_val: Optional minimum integer value (defaults to profile minimum).
    max_val: Optional maximum integer value (defaults to profile maximum).
    seed: Random number generator seed.

  Returns:
    NumPy array with integer test values and injected boundary extremes.

  Raises:
    KeyError: If dtype_str is not in INTEGER_PROFILES.
  """
  if dtype_str not in INTEGER_PROFILES:
    raise KeyError(
        f"Unsupported integer dtype_str '{dtype_str}'. Supported:"
        f" {list(INTEGER_PROFILES.keys())}"
    )
  prof = INTEGER_PROFILES[dtype_str]
  low = min_val if min_val is not None else prof.min_val
  high = max_val if max_val is not None else prof.max_val
  rng = np.random.default_rng(seed)
  arr = rng.integers(
      low, high, size=shape, endpoint=True, dtype=prof.numpy_dtype
  )
  if arr.size >= 4:
    flat = arr.reshape(-1)
    flat[0] = low
    flat[1] = high
    flat[2] = 0 if low <= 0 <= high else low
    flat[3] = -1 if low <= -1 <= high else (1 if low <= 1 <= high else low)
    arr = flat.reshape(shape)
  return arr


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

  if dtype_str == "bool":
    suite = []
    # 1. Causal mask batch
    causal_tensors = []
    for s in shapes:
      if len(s) >= 2:
        causal_tensors.append(
            _convert(
                generate_mask_tensor(
                    s, mask_type="causal", dtype_str="bool", seed=seed
                )
            )
        )
      else:
        causal_tensors.append(
            _convert(
                generate_mask_tensor(
                    s,
                    mask_type="bernoulli",
                    density=0.5,
                    dtype_str="bool",
                    seed=seed,
                )
            )
        )
    suite.append({
        "name": "causal_masks",
        "args": tuple(causal_tensors),
        "kwargs": {},
        "regime": "causal_mask",
    })

    # 2. Bernoulli sparse mask batches
    densities = [0.1, 0.5, 0.9] if tier != "fast_agent" else [0.5]
    for idx, d in enumerate(densities):
      sparse_tensors = [
          _convert(
              generate_mask_tensor(
                  s,
                  mask_type="bernoulli",
                  density=d,
                  dtype_str="bool",
                  seed=seed + 10 + idx * 5 + i,
              )
          )
          for i, s in enumerate(shapes)
      ]
      suite.append({
          "name": f"bernoulli_density_{int(d*100)}pct",
          "args": tuple(sparse_tensors),
          "kwargs": {},
          "regime": "bernoulli_mask",
      })

    # 3. Padding mask batch
    padding_tensors = [
        _convert(
            generate_mask_tensor(
                s,
                mask_type="padding",
                dtype_str="bool",
                seed=seed + 100 + i,
            )
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": "padding_masks",
        "args": tuple(padding_tensors),
        "kwargs": {},
        "regime": "padding_mask",
    })
    return suite

  if dtype_str in INTEGER_PROFILES:
    prof = INTEGER_PROFILES[dtype_str]
    suite = []
    # 1. Small dynamic range batch (e.g. [-10, 20] or [0, 20])
    low_val = -10 if prof.is_signed else 0
    small_tensors = [
        _convert(
            generate_integer_tensor(
                s,
                dtype_str=dtype_str,
                min_val=low_val,
                max_val=20,
                seed=seed + i,
            )
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": "small_dynamic_range",
        "args": tuple(small_tensors),
        "kwargs": {},
        "regime": "small_int",
    })

    # 2. Bounded index batches (e.g. expert IDs / vocab indices [0, 64])
    index_tensors = [
        _convert(
            generate_index_tensor(
                s, upper_bound=64, dtype_str=dtype_str, seed=seed + 20 + i
            )
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": "bounded_indices_64",
        "args": tuple(index_tensors),
        "kwargs": {},
        "regime": "bounded_index",
    })

    # 3. Monotonic segment IDs batch
    seg_tensors = [
        _convert(
            generate_segment_ids_tensor(
                s,
                num_segments=min(8, s[-1] if s else 1),
                is_sorted=True,
                dtype_str=dtype_str,
                seed=seed + 30 + i,
            )
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": "monotonic_segment_ids",
        "args": tuple(seg_tensors),
        "kwargs": {},
        "regime": "segment_ids",
    })

    # 4. Full boundary extremes batch (min_int, max_int, 0, 1, -1)
    boundary_tensors = [
        _convert(
            generate_integer_tensor(s, dtype_str=dtype_str, seed=seed + 40 + i)
        )
        for i, s in enumerate(shapes)
    ]
    suite.append({
        "name": "boundary_extremes",
        "args": tuple(boundary_tensors),
        "kwargs": {},
        "regime": "boundary_extremes",
    })
    return suite

  if dtype_str not in PROFILES:
    raise KeyError(
        f"Unsupported dtype_str '{dtype_str}'. Supported:"
        f" {sorted(list(ALL_SUPPORTED_DTYPES))}"
    )

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
      "fp8_e5m2", "int32", "int64", "int16", "int8", "uint32", "uint8", "bool").
    tier: Test thoroughness tier ("fast_agent", "presubmit", "deep_fuzzing").
    seed: Random number generator seed.
    persisted_path: Optional path to a .npz file or resource to load/save.
    mode: Persistence mode ("auto", "read_only", "record").
    as_jax_arrays: If True, returns arrays as jax.Array (jnp.ndarray).

  Returns:
    List of test batch dicts with 'name', 'args', 'kwargs', and 'regime'.

  Raises:
    RuntimeError: If mode is 'read_only' and persisted_path does not exist.
    KeyError: If dtype_str is not supported.
  """
  if dtype_str not in ALL_SUPPORTED_DTYPES:
    raise KeyError(
        f"Unsupported dtype_str '{dtype_str}'. Supported:"
        f" {sorted(list(ALL_SUPPORTED_DTYPES))}"
    )
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
