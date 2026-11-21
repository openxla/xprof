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
"""Reports how far a candidate's outputs are from the reference, in ULPs.

The report is for information only. It never decides whether an output is
correct: the item's `atol` and `rtol` do that, through `np.allclose`. A ULP
(unit in the last place) count shows how close a passing output is, and how
far off a failing one is, in the units of the output's own dtype.

Each output leaf is measured in its reference's dtype, against the ULP
contract that xparity gives for that dtype. NaN and infinity have no useful
ULP distance, so they are counted instead and left out of the distances.

This module needs only NumPy and the xparity `ulp` module. It does not import
JAX: the caller gives it host arrays.
"""

from __future__ import annotations

import collections.abc
import logging
import math
from typing import Any

import numpy as np

from xprof.xparity import ulp

_Sequence = collections.abc.Sequence

logger = logging.getLogger(__name__)

# Report status values.
STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_NOT_MEASURED = "not_measured"
STATUS_ERROR = "error"

# Counts of NaN and infinite elements, summed from the leaves into the report.
_NON_FINITE_KEYS = (
    "nan_count",
    "inf_count",
    "reference_nan_count",
    "reference_inf_count",
    "non_finite_mismatch_count",
)


def not_measured(detail: str) -> dict[str, Any]:
  """Returns a report for outputs that cannot be compared at all.

  Args:
    detail: Why the outputs were not measured.
  """
  return {"status": STATUS_NOT_MEASURED, "detail": detail}


def _kind(dtype: np.dtype) -> str | None:
  """Returns "float", "int" or "bool", or None if xparity cannot measure it."""
  canonical = ulp.resolve_canonical_dtype(dtype)
  if canonical == "bool":
    return "bool"
  if canonical in ulp.INTEGER_DTYPES:
    return "int"
  # These are the floating-point dtypes that have a ULP bit layout.
  if canonical in ulp.DTYPE_TOLERANCES:
    return "float"
  return None


def _distance(
    candidate: np.ndarray, reference: np.ndarray, kind: str, dtype_name: str
) -> np.ndarray:
  """Returns the elementwise ULP distance, in the reference's dtype."""
  if kind == "int":
    # Integers narrower than 64 bits are widened, so a difference cannot wrap.
    if candidate.dtype.itemsize < 8:
      candidate = candidate.astype(np.int64)
    if reference.dtype.itemsize < 8:
      reference = reference.astype(np.int64)
    dtype_name = str(reference.dtype)
  return np.asarray(
      ulp.compute_ulp_distance(candidate, reference, dtype_name),
      dtype=np.int64,
  )


def _scalar(value: Any) -> float | int | bool:
  """Converts one element to a JSON-friendly Python scalar."""
  item = value.item() if isinstance(value, np.generic) else value
  if isinstance(item, (bool, int)):
    return item
  return float(item)


def _measure_leaf(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
  """Measures one output leaf against its reference leaf.

  Args:
    candidate: Candidate leaf, on the host.
    reference: Reference leaf, on the host.

  Returns:
    A leaf record. `status` is "ok" when the leaf was measured, else one of
    "shape_mismatch", "unsupported_dtype" or "dtype_mismatch". A measured
    record also holds the distances of its finite elements, under the private
    key `_distances`.
  """
  dtype_name = ulp.resolve_canonical_dtype(reference.dtype)
  record: dict[str, Any] = {
      "dtype": dtype_name,
      "output_dtype": ulp.resolve_canonical_dtype(candidate.dtype),
      "elements": int(reference.size),
  }
  if candidate.shape != reference.shape:
    record["status"] = "shape_mismatch"
    record["detail"] = f"{candidate.shape} vs {reference.shape}"
    return record
  kind = _kind(reference.dtype)
  candidate_kind = _kind(candidate.dtype)
  if kind is None or candidate_kind is None:
    record["status"] = "unsupported_dtype"
    record["detail"] = (
        f"cannot measure {record['output_dtype']} against {dtype_name} in ULPs"
    )
    return record
  if kind != candidate_kind:
    record["status"] = "dtype_mismatch"
    record["detail"] = f"{candidate_kind} output vs {kind} reference"
    return record

  contract_ulp, _ = ulp.get_contract(dtype_name)
  record["contract_ulp"] = contract_ulp
  distance = _distance(candidate, reference, kind, dtype_name)
  same_dtype = candidate.dtype == reference.dtype
  if same_dtype:
    record["bit_mismatch_count"] = int(
        np.count_nonzero(ulp.bitwise_mismatch_mask(candidate, reference))
    )
  else:
    # Bit patterns of different dtypes are never identical.
    record["bit_mismatch_count"] = int(candidate.size)

  finite = np.ones(reference.shape, dtype=bool)
  record.update(dict.fromkeys(_NON_FINITE_KEYS, 0))
  if kind == "float":
    # Each array is checked in its own dtype. A cast to a narrower float
    # would turn large finite values infinite.
    candidate_nan, candidate_inf = np.isnan(candidate), np.isinf(candidate)
    reference_nan, reference_inf = np.isnan(reference), np.isinf(reference)
    if not same_dtype:
      # The candidate is measured in the reference's dtype. A value past the
      # range of that dtype has no meaningful distance.
      with np.errstate(over="ignore", invalid="ignore"):
        narrowed = candidate.astype(reference.dtype)
      candidate_nan |= np.isnan(narrowed)
      candidate_inf |= np.isinf(narrowed)
    non_finite = candidate_nan | candidate_inf | reference_nan | reference_inf
    same_non_finite = (candidate_nan & reference_nan) | (
        candidate_inf
        & reference_inf
        & (np.signbit(candidate) == np.signbit(reference))
    )
    record.update(
        nan_count=int(np.count_nonzero(candidate_nan)),
        inf_count=int(np.count_nonzero(candidate_inf)),
        reference_nan_count=int(np.count_nonzero(reference_nan)),
        reference_inf_count=int(np.count_nonzero(reference_inf)),
        non_finite_mismatch_count=int(
            np.count_nonzero(non_finite & ~same_non_finite)
        ),
    )
    finite = ~non_finite
    distance = np.where(finite, distance, 0)

  record["status"] = "ok"
  record["max_ulp"] = int(distance.max()) if distance.size else 0
  record["over_contract_count"] = int(np.count_nonzero(distance > contract_ulp))
  if record["max_ulp"] > 0:
    worst = int(np.argmax(distance))
    record["worst_index"] = [
        int(i) for i in np.unravel_index(worst, reference.shape)
    ]
    record["worst_reference_value"] = _scalar(reference.reshape(-1)[worst])
    record["worst_output_value"] = _scalar(candidate.reshape(-1)[worst])
  record["_distances"] = distance[finite]
  return record


def compare_outputs(
    reference_samples: _Sequence[_Sequence[np.ndarray]],
    candidate_samples: _Sequence[_Sequence[np.ndarray]],
) -> dict[str, Any]:
  """Reports the ULP distance of a candidate's outputs from the reference.

  Never raises: a failure is reported with status "error".

  Args:
    reference_samples: Per sample, the flattened output leaves of the reference,
      as host arrays.
    candidate_samples: The same for the candidate, in the same order.

  Returns:
    A JSON-friendly report with these keys:
      status: "ok" if every leaf was measured, "partial" if only some were,
        "not_measured" if none were, or "error".
      detail: Why a leaf was not measured, or why the report failed.
      samples: Number of samples compared.
      leaves_total, leaves_measured: Leaves over all samples.
      elements: Elements in the measured leaves.
      max_ulp: Largest distance over the finite elements of measured leaves.
      median_ulp: Median distance over the same elements.
      contract_ulp: The xparity contract of the leaf with `max_ulp`.
      within_contract: No measured element is over its leaf's contract.
      bit_identical: Every leaf was measured and matches bit for bit.
      dtype: The reference dtype of the leaf with `max_ulp`.
      dtype_mismatch_count: Leaves whose candidate dtype differs from the
        reference dtype. Float leaves are still measured, in the reference
        dtype.
      nan_count, inf_count: NaN and infinite elements in the candidate.
      reference_nan_count, reference_inf_count: The same in the reference.
      non_finite_mismatch_count: Elements where the candidate and the
        reference do not have the same NaN or infinity.
      worst_sample, worst_leaf, worst_index: Where `max_ulp` occurs. Absent
        when `max_ulp` is 0.
      worst_reference_value, worst_output_value: The values there.
  """
  try:
    return _compare_outputs(reference_samples, candidate_samples)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.exception("ULP report failed")
    return {"status": STATUS_ERROR, "detail": f"{type(e).__name__}: {e}"}


def _compare_outputs(
    reference_samples: _Sequence[_Sequence[np.ndarray]],
    candidate_samples: _Sequence[_Sequence[np.ndarray]],
) -> dict[str, Any]:
  """Implements `compare_outputs`, which handles any exception."""
  if len(reference_samples) != len(candidate_samples):
    return not_measured(
        f"{len(candidate_samples)} candidate samples vs"
        f" {len(reference_samples)} reference samples"
    )
  if not reference_samples:
    return not_measured("no samples")

  report: dict[str, Any] = {
      "samples": len(reference_samples),
      "leaves_total": 0,
      "leaves_measured": 0,
      "elements": 0,
      "max_ulp": 0,
      "within_contract": True,
      "bit_identical": True,
      "dtype_mismatch_count": 0,
  }
  report.update(dict.fromkeys(_NON_FINITE_KEYS, 0))
  distances: list[np.ndarray] = []
  details: list[str] = []
  worst: tuple[int, int, dict[str, Any]] | None = None

  for sample, (reference_leaves, candidate_leaves) in enumerate(
      zip(reference_samples, candidate_samples)
  ):
    if len(reference_leaves) != len(candidate_leaves):
      return not_measured(
          f"sample {sample}: {len(candidate_leaves)} candidate leaves vs"
          f" {len(reference_leaves)} reference leaves"
      )
    for leaf, (reference, candidate) in enumerate(
        zip(reference_leaves, candidate_leaves)
    ):
      reference = np.asarray(reference)
      candidate = np.asarray(candidate)
      report["leaves_total"] += 1
      if candidate.dtype != reference.dtype:
        report["dtype_mismatch_count"] += 1
      record = _measure_leaf(candidate, reference)
      if record["status"] != "ok":
        report["bit_identical"] = False
        details.append(
            f"sample {sample} leaf {leaf}: {record['status']}"
            f" ({record['detail']})"
        )
        continue
      report["leaves_measured"] += 1
      report["elements"] += record["elements"]
      for name in _NON_FINITE_KEYS:
        report[name] += record[name]
      if record["bit_mismatch_count"]:
        report["bit_identical"] = False
      if record["over_contract_count"]:
        report["within_contract"] = False
      distances.append(record["_distances"])
      if worst is None or record["max_ulp"] > worst[2]["max_ulp"]:
        worst = (sample, leaf, record)

  if worst is None:
    report["status"] = STATUS_NOT_MEASURED
    report["within_contract"] = None
    report["bit_identical"] = None
    report["max_ulp"] = None
    report["detail"] = "; ".join(details[:5])
    return report

  report["status"] = STATUS_PARTIAL if details else STATUS_OK
  if details:
    shown = "; ".join(details[:5])
    more = len(details) - 5
    report["detail"] = shown + (f"; and {more} more" if more > 0 else "")
  sample, leaf, record = worst
  report["max_ulp"] = record["max_ulp"]
  report["contract_ulp"] = record["contract_ulp"]
  report["dtype"] = record["dtype"]
  all_distances = np.concatenate(distances)
  median = float(np.median(all_distances)) if all_distances.size else 0.0
  report["median_ulp"] = median if math.isfinite(median) else None
  if record["max_ulp"] > 0:
    report["worst_sample"] = sample
    report["worst_leaf"] = leaf
    for name in (
        "worst_index",
        "worst_reference_value",
        "worst_output_value",
    ):
      report[name] = record[name]
  return report
