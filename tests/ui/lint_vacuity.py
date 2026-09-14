"""Gate 1: static AST vacuity analysis for the UI test suites."""

import ast
import os
import pathlib
import sys

# Names of calls that count as an assertion when they appear in a test body.
_ASSERTION_NAMES = frozenset({"expect"})
_ASSERTION_PREFIXES = ("assert", "_assert")
# Keyword arguments through which a Playwright locator receives its selector.
_SELECTOR_KEYWORDS = frozenset({"selector", "selector_or_locator"})


def get_default_scan_dir() -> pathlib.Path:
  """Returns the directory holding the UI test sources."""
  test_srcdir = os.environ.get("TEST_SRCDIR")
  test_workspace = os.environ.get("TEST_WORKSPACE", "google3")
  if test_srcdir:
    runfiles_dir = (
        pathlib.Path(test_srcdir)
        / test_workspace
        / "third_party/xprof/tests/ui"
    )
    if runfiles_dir.is_dir():
      return runfiles_dir
  return pathlib.Path(__file__).parent


def _is_assertion(node: ast.AST) -> bool:
  """Returns whether the node asserts directly or calls an assertion helper."""
  if isinstance(node, ast.Assert):
    return True
  if not isinstance(node, ast.Call):
    return False
  if isinstance(node.func, ast.Name):
    name = node.func.id
  elif isinstance(node.func, ast.Attribute):
    name = node.func.attr
  else:
    # The callee is computed (a subscript or another call), so there is no
    # name to match against.
    return False
  return name in _ASSERTION_NAMES or name.startswith(_ASSERTION_PREFIXES)


def _is_blank_literal(node: ast.AST) -> bool:
  """Returns whether the node is a string literal with no visible content."""
  return (
      isinstance(node, ast.Constant)
      and isinstance(node.value, str)
      and not node.value.strip()
  )


def _check_locator_call(call: ast.Call, func_name: str) -> list[str]:
  """Returns violations for locator calls built from a blank selector."""
  if not (isinstance(call.func, ast.Attribute) and call.func.attr == "locator"):
    return []
  violations = []
  if call.args and _is_blank_literal(call.args[0]):
    violations.append(f"Empty or whitespace locator in '{func_name}'.")
  for keyword in call.keywords:
    if keyword.arg in _SELECTOR_KEYWORDS and _is_blank_literal(keyword.value):
      violations.append(
          f"Empty or whitespace locator keyword '{keyword.arg}' in"
          f" '{func_name}'."
      )
  return violations


def check_vacuity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
  """Returns the vacuity violations found in a single function definition."""
  violations = []
  assertions = 0
  for child in ast.walk(node):
    if isinstance(child, (ast.If, ast.IfExp)):
      # An assertion reachable only through a branch silently passes whenever
      # that branch is not taken.
      if any(_is_assertion(inner) for inner in ast.walk(child)):
        kind = "ternary" if isinstance(child, ast.IfExp) else "if block"
        violations.append(
            f"Conditional assertion ({kind}) detected in '{node.name}'."
        )
    elif _is_assertion(child):
      assertions += 1
    elif isinstance(child, ast.Call):
      violations.extend(_check_locator_call(child, node.name))
  if not assertions and node.name.startswith("test"):
    violations.append(f"Function '{node.name}' has 0 assertions or expects.")
  return violations


def lint_file(file_path: pathlib.Path) -> list[tuple[str, int, str]]:
  """Returns the violations found in a single Python test file."""
  try:
    source = file_path.read_text(encoding="utf-8")
  except (OSError, UnicodeDecodeError) as e:
    return [(str(file_path), 0, f"Unreadable file: {e}")]
  try:
    tree = ast.parse(source, filename=str(file_path))
  except SyntaxError as e:
    return [(str(file_path), e.lineno or 0, f"SyntaxError: {e.msg}")]
  except (RecursionError, ValueError) as e:
    # Deeply nested or NUL-bearing sources defeat the parser itself; report
    # them as violations instead of aborting the whole run.
    return [(str(file_path), 0, f"Unparsable file: {e}")]

  results = []
  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      for err in check_vacuity(node):
        results.append((str(file_path), node.lineno, err))
  return results


def collect_files(paths: list[pathlib.Path]) -> list[pathlib.Path]:
  """Returns the deduplicated, ordered test files found under the paths."""
  files: set[pathlib.Path] = set()
  for path in paths:
    if path.is_file():
      files.add(path)
    else:
      files.update(path.rglob("test_*.py"))
  return sorted(files)


def main() -> int:
  """Runs the linter over the requested paths and reports the verdict."""
  paths = [pathlib.Path(p) for p in sys.argv[1:]] or [get_default_scan_dir()]
  files = collect_files(paths)

  if not files:
    print(
        "Gate 1 Vacuity Linter: FAILED (0 test files discovered; hermetic"
        " sandbox runfiles vacuity detected)."
    )
    return 1

  violations = []
  for f in files:
    violations.extend(lint_file(f))

  if not violations:
    print(
        f"Gate 1 Vacuity Linter: All {len(files)} test file(s) passed cleanly"
        " (0 violations)."
    )
    return 0

  print(f"Gate 1 Vacuity Linter: Found {len(violations)} violation(s):")
  for path, line, msg in violations:
    print(f"  {path}:{line}: [VACUITY] {msg}")
  return 1


if __name__ == "__main__":
  sys.exit(main())
