"""Test reading TPU counter IDs resources."""
import importlib.resources

try:
  # Try to read using importlib.resources
  content = importlib.resources.read_text(
      "google3.third_party.xprof.utils", "tpu_counter_ids_v7x.h"
  )
  print("Success with importlib.resources!")
  print(content[:100])
except ImportError as e:
  print(f"Failed with importlib.resources: {e}")

try:
  # Try with files()
  ref = (
      importlib.resources.files("google3.third_party.xprof.utils")
      / "tpu_counter_ids_v7x.h"
  )
  content = ref.read_text()
  print("Success with importlib.resources.files!")
  print(content[:100])
except ImportError as e:
  print(f"Failed with importlib.resources.files: {e}")
