"""Utility functions for xprof integration tests."""

import copy
import difflib
import gzip
import json
import os
import urllib.error
import urllib.request
from absl import logging
import pytest

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
BLUE = "\033[94m"


def print_json_diff(actual, expected):
  """Generates and prints a colored Git-style diff between two JSON objects."""
  actual_formatted = json.dumps(actual, indent=2, sort_keys=True)
  expected_formatted = json.dumps(expected, indent=2, sort_keys=True)

  diff = difflib.unified_diff(
      expected_formatted.splitlines(),
      actual_formatted.splitlines(),
      fromfile="Expected (Golden)",
      tofile="Actual (Server)",
      lineterm="",
  )

  diff_output = []
  for line in diff:
    if line.startswith("+"):
      diff_output.append(f"{GREEN}{line}{RESET}")
    elif line.startswith("-"):
      diff_output.append(f"{RED}{line}{RESET}")
    elif line.startswith("^"):
      diff_output.append(f"{BLUE}{line}{RESET}")
    else:
      diff_output.append(line)

  # Join and return (or print directly)
  return "\n".join(diff_output)


def prune_json(data: any, ignore_paths: list[str]) -> any:
  """Removes specified paths from a JSON object.

  Supports wildcards '*' for both Lists and Dictionaries (iterating over
  values).

  Args:
    data: The JSON object (dict or list) to prune.
    ignore_paths: A list of strings, where each string is a dot-separated path
      to a key or index to be removed. Wildcards '*' are supported.

  Returns:
    A deep copy of the input `data` with the specified paths removed.
  """
  if not ignore_paths:
    return data

  data_copy = copy.deepcopy(data)

  def _recursive_delete(current, keys):
    if not keys:
      return

    key = keys[0]
    is_last = len(keys) == 1

    # --- HANDLE WILDCARD (*) ---
    if key == "*":
      # Case A: Current node is a List -> Iterate items
      if isinstance(current, list):
        for item in current:
          _recursive_delete(item, keys[1:])
      # Case B: Current node is a Dict -> Iterate values
      elif isinstance(current, dict):
        for k in current:
          _recursive_delete(current[k], keys[1:])
      return

    # --- HANDLE EXACT KEY MATCH ---
    if isinstance(current, dict) and key in current:
      if is_last:
        del current[key]
      else:
        _recursive_delete(current[key], keys[1:])

  for path in ignore_paths:
    _recursive_delete(data_copy, path.split("."))

  return data_copy


def verify_golden(
    actual_data: any, golden_filename: str, ignore_paths: list[str] = None
) -> None:
  """Compares actual_data against a stored .json.gz file."""

  base_dir = os.path.dirname(__file__)
  # Automatically expect a gzipped file
  if not golden_filename.endswith(".gz"):
    golden_filename += ".gz"

  golden_path = os.path.join(base_dir, "data", "golden", golden_filename)

  # Prune actual data
  clean_actual = prune_json(actual_data, ignore_paths)

  # --- Mode 1: Update Goldens (Write Compressed) ---
  if os.environ.get("UPDATE_GOLDENS") == "1":
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    # 'wt' = Write Text mode (handles compression transparently)
    with gzip.open(golden_path, "wt", encoding="utf-8") as f:
      json.dump(clean_actual, f, indent=2, sort_keys=True)
      f.write("\n")
    logging.warning("Updated golden file: %s", golden_path)
    return

  # --- Mode 2: Verify Goldens (Read Compressed) ---
  if not os.path.exists(golden_path):
    pytest.fail(f"Golden file missing: {golden_filename}")

  try:
    # 'rt' = Read Text mode
    with gzip.open(golden_path, "rt", encoding="utf-8") as f:
      expected_data = json.load(f)
  except (OSError, gzip.BadGzipFile):
    pytest.fail(
        f"Could not decompress golden file: {golden_path}. Is it a valid .gz"
        " file?"
    )

  # Prune expected data
  clean_expected = prune_json(expected_data, ignore_paths)

  if clean_actual != clean_expected:
    diff_text = print_json_diff(clean_actual, clean_expected)
    pytest.fail(
        f"Data mismatch against golden: {golden_filename}\n\n"
        "--- DIFF START ---\n"
        f"{diff_text}\n"
        "--- DIFF END ---"
    )


def fetch_text(url: str) -> str:
  """Fetches URL and returns text content, handling transparent GZIP decompression."""
  req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})

  try:
    with urllib.request.urlopen(req, timeout=30) as response:
      if response.status != 200:
        pytest.fail(f"HTTP {response.status} error from {url}")
      raw_data = response.read()
      if response.info().get("Content-Encoding") == "gzip":
        raw_data = gzip.decompress(raw_data)
      return raw_data.decode("utf-8")
  except urllib.error.URLError as e:
    pytest.fail(f"Failed to fetch {url}: {e}")
  except UnicodeDecodeError as e:
    pytest.fail(f"Invalid or undecompressible data received from {url}: {e}")


def fetch_json(url: str) -> any:
  """Fetches URL and returns parsed JSON, handling transparent GZIP decompression."""
  req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})

  try:
    with urllib.request.urlopen(req, timeout=10) as response:
      if response.status != 200:
        pytest.fail(f"HTTP {response.status} error from {url}")

      raw_data = response.read()

      if response.info().get("Content-Encoding") == "gzip":
        raw_data = gzip.decompress(raw_data)

      return json.loads(raw_data.decode("utf-8"))
  except urllib.error.URLError as e:
    pytest.fail(f"Failed to fetch {url}: {e}")
  except (json.JSONDecodeError, UnicodeDecodeError) as e:
    pytest.fail(f"Invalid or undecompressible data received from {url}: {e}")


def get_available_runs() -> list[str]:
  """Scans the integration_tests/data/test_xplanes/plugins/profile/ directory.

  Returns:
    A list of all subdirectory names (which represent runs).
  """
  base_dir = os.path.dirname(__file__)
  profile_dir = os.path.join(
      base_dir, "data", "test_xplanes", "plugins", "profile"
  )

  if not os.path.isdir(profile_dir):
    return []

  return [
      d
      for d in os.listdir(profile_dir)
      if os.path.isdir(os.path.join(profile_dir, d))
  ]
