"""Defines the counter extractor routine for TPU counters from v7x."""

import os
import re


def get_all_counters(device_type):
  """Parses tpu_counter_ids_{device_type}.h and returns a list of counters with names and indices."""
  content = None
  filename = f'tpu_counter_ids_{device_type}.h'

  if content is None:
    dirname = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirname, '..', '..', '..', 'utils', filename)

    if not os.path.exists(filepath):
      raise FileNotFoundError('Counter file not found at ' + filepath)

    with open(filepath, 'r') as f:
      content = f.read()

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
      name, val_str = line.split('=')
      name = name.strip()
      val = int(val_str.strip())
      if not name.startswith('//'):
        counters.append({'name': name.lower(), 'val': val})

  return counters
