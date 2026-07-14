"""Patches dependencies in node_modules to prevent build errors.

This script modifies node_modules packages to work around known issues:
1. Replaces a duplicate link error in @material/theme/_keys.scss with a return.
2. Adds a null-guard in typescript/lib files to prevent TypeError in
   addResolutionDiagnostics when concatjs compilerHost returns undefined.

Usage:
  python patch_keys.py
"""

import pathlib
import re
import sys

_SEARCH_STR = "@error '#{$key} already has a link';"
_REPLACE_STR = "@return $key;"


def _patch_file(file_path: pathlib.Path) -> None:
  """Patches the target file to replace the error string with a return string."""
  try:
    with open(file_path, "r", encoding="utf-8") as target_file:
      original_content = target_file.read()
  except FileNotFoundError:
    print(f"Warning: {file_path} not found")
    return

  if _SEARCH_STR not in original_content:
    print(f"Warning: Search string not found in {file_path}. Already patched?")
    return

  patched_content = original_content.replace(_SEARCH_STR, _REPLACE_STR)

  with open(file_path, "w", encoding="utf-8") as target_file:
    target_file.write(patched_content)
  print(f"Successfully patched {file_path}")


def _patch_typescript(node_modules_dir: pathlib.Path) -> None:
  """Patches TypeScript 5.4 addResolutionDiagnostics TypeError crash."""
  ts_lib = node_modules_dir / "typescript" / "lib"
  search_ts_1 = "function addResolutionDiagnostics(resolution) {"
  replace_ts_1 = (
      "function addResolutionDiagnostics(resolution) { if (!resolution ||"
      " !resolution.resolutionDiagnostics) return;"
  )
  search_ts_2 = "const { resolvedTypeReferenceDirective } = resolution;"
  replace_ts_2 = (
      "if (!resolution) return; const { resolvedTypeReferenceDirective } ="
      " resolution;"
  )
  search_ts_3 = "resolution.resolvedTypeReferenceDirective"
  replace_ts_3 = "(resolution && resolution.resolvedTypeReferenceDirective)"
  for name in [
      "typescript.js",
      "tsc.js",
      "tsserver.js",
      "typescriptServices.js",
  ]:
    ts_file = ts_lib / name
    if ts_file.exists():
      try:
        with open(ts_file, "r", encoding="utf-8") as target_file:
          content = target_file.read()
        patched = content
        if search_ts_1 in patched:
          patched = patched.replace(search_ts_1, replace_ts_1)
        if search_ts_2 in patched:
          patched = patched.replace(search_ts_2, replace_ts_2)
        if search_ts_3 in patched:
          patched = patched.replace(search_ts_3, replace_ts_3)
        if patched != content:
          with open(ts_file, "w", encoding="utf-8") as target_file:
            target_file.write(patched)
          print(f"Successfully patched {ts_file}")
      except OSError as e:
        print(f"Warning: Failed to patch {ts_file}: {e}")


def _patch_rollup(node_modules_dir: pathlib.Path) -> None:
  """Patches Rollup default ecmaVersion to support ES2022 class static blocks."""
  target_files = [
      node_modules_dir / "rollup" / "dist" / "shared" / "rollup.js",
      node_modules_dir / "rollup" / "dist" / "rollup.js",
  ]
  for target_file in target_files:
    if target_file.exists():
      try:
        with open(target_file, "r", encoding="utf-8") as f:
          content = f.read()
        patched = content.replace(
            "ecmaVersion = 2020", "ecmaVersion = 'latest'"
        )
        patched = patched.replace(
            "ecmaVersion: 2020", "ecmaVersion: 'latest'"
        )
        if patched != content:
          with open(target_file, "w", encoding="utf-8") as f:
            f.write(patched)
          print(f"Successfully patched {target_file}")
      except OSError as e:
        print(f"Warning: Failed to patch {target_file}: {e}")


def _transform_static_blocks(code: str) -> str:
  """Transforms ES2022 static blocks into static field initializers."""
  idx = 0
  result = []
  while True:
    match = re.search(r"\bstatic\s*\{", code[idx:])
    if not match:
      result.append(code[idx:])
      break
    start = idx + match.start()
    brace_start = idx + match.end() - 1
    result.append(code[idx:start])
    result.append("static __init = (() => {")
    depth = 1
    pos = brace_start + 1
    while pos < len(code) and depth > 0:
      if code[pos] == "{":
        depth += 1
      elif code[pos] == "}":
        depth -= 1
      pos += 1
    if depth == 0:
      inner = code[brace_start + 1 : pos - 1]
      result.append(inner)
      result.append("})();")
      idx = pos
    else:
      result.append(code[brace_start:])
      break
  return "".join(result)


def _patch_static_blocks(node_modules_dir: pathlib.Path) -> None:
  """Transforms class static initialization blocks across all node_modules for Rollup."""
  if not node_modules_dir.exists():
    return
  for mjs_file in node_modules_dir.glob("**/*.mjs"):
    try:
      with open(mjs_file, "r", encoding="utf-8") as f:
        content = f.read()
      if "static {" in content or "static\n{" in content:
        patched = _transform_static_blocks(content)
        if patched != content:
          with open(mjs_file, "w", encoding="utf-8") as f:
            f.write(patched)
          print(f"Successfully patched static blocks in {mjs_file}")
    except OSError as e:
      print(f"Warning: Failed to patch {mjs_file}: {e}")


def main() -> None:
  if len(sys.argv) > 1:
    print("Error: Too many command-line arguments. Usage: python patch_keys.py")
    sys.exit(1)

  node_modules_dir = pathlib.Path(__file__).resolve().parent / "node_modules"
  target = node_modules_dir / "@material" / "theme" / "_keys.scss"
  _patch_file(target)
  _patch_typescript(node_modules_dir)
  _patch_rollup(node_modules_dir)
  _patch_static_blocks(node_modules_dir)


if __name__ == "__main__":
  main()
