"""Patches @material/theme/_keys.scss to prevent build errors.

This script modifies the _keys.scss file within node_modules to replace
a specific error-throwing line with a return statement, working around
a build issue where duplicate links cause Sass compilation to fail.

Usage:
  python patch_keys.py
"""

import pathlib
import sys

_SEARCH_STR = "@error '#{$key} already has a link';"
_REPLACE_STR = "@return $key;"


def _patch_file(file_path: pathlib.Path) -> None:
  """Patches the target file to replace the error string with a return string.

  Args:
    file_path: The path to the file to be patched.

  Raises:
    OSError: If reading or writing the file fails for reasons other than
      FileNotFoundError.
  """
  try:
    with open(file_path, "r", encoding="utf-8") as f:
      original_content = f.read()
  except FileNotFoundError:
    print(f"Warning: {file_path} not found")
    return

  if _SEARCH_STR not in original_content:
    print(f"Warning: Search string not found in {file_path}. Already patched?")
    return

  patched_content = original_content.replace(_SEARCH_STR, _REPLACE_STR)

  with open(file_path, "w", encoding="utf-8") as f:
    f.write(patched_content)
  print(f"Successfully patched {file_path}")


if __name__ == "__main__":
  if len(sys.argv) > 1:
    print("Error: Too many command-line arguments. Usage: python patch_keys.py")
    sys.exit(1)

  target = (
      pathlib.Path(__file__).resolve().parent
      / "node_modules"
      / "@material"
      / "theme"
      / "_keys.scss"
  )
  _patch_file(target)
