"""Defines the counter extractor routine for TPU counters."""

import functools
import importlib.resources
import re

VALID_DEVICE_TYPES = frozenset({'v6e', 'v7x'})


@functools.lru_cache()
def get_all_counters(device_type: str) -> list[dict[str, int]]:
  """Parses tpu_counter_ids_{device_type}.h and returns a list of counters with names and indices."""
  if device_type not in VALID_DEVICE_TYPES:
    raise ValueError(f'Unsupported device_type: {device_type}')
  filename = f'tpu_counter_ids_{device_type}.h'

  try:
    ref = importlib.resources.files('xprof') / 'utils' / filename
  except ImportError:
    pass
  content = ref.read_text(encoding='utf-8')
  # Remove C++ style line comments from content.
  content = re.sub(r'//.*', '', content)

  # The counter file is a C++ header file with an enum.
  # Example:
  # enum TpuCounterIdsTpu{version} : uint64_t {
  #   TPU_COUNTER_ID_FOO = 12345,
  #   TPU_COUNTER_ID_BAR = 67890,
  #   ...
  # };
  match = re.search(r'enum\s+\w+\s*:\s*uint64_t\s*\{([^}]+)\}', content)
  if not match:
    return []

  enum_body = match.group(1)
  counters = []
  for line in enum_body.split(','):
    line = line.strip()
    if '=' in line:
      name, val_str = line.split('=', 1)
      try:
        name = name.strip()
        val = int(val_str.strip(), 0)
        counters.append({'name': name.lower(), 'val': val})
      except ValueError:
        continue

  return counters
