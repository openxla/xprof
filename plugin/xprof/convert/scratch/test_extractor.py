"""Test reading TPU counter IDs resources."""
from xprof.convert import counter_extractor

try:
  counters = counter_extractor.get_all_counters("v7x")
  print(f"Success! Found {len(counters)} counters.")
except (ImportError, ValueError) as e:
  print(f"Failed: {e}")
