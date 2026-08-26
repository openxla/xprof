"""Gate 1: Static AST Vacuity Linter for Playwright Test Suites."""

import ast
import pathlib
import sys
from typing import List, Tuple


def check_vacuity(node: ast.FunctionDef) -> List[str]:
  """Checks a single test function for anti-patterns."""
  errors = []
  asserts = 0
  for child in ast.walk(node):
    if isinstance(child, ast.If):
      if any(isinstance(n, ast.Assert) for n in ast.walk(child)):
        errors.append(f"Conditional assertion detected in '{node.name}'.")
    elif isinstance(child, ast.Assert):
      asserts += 1
    elif isinstance(child, ast.Call):
      if (isinstance(child.func, ast.Name) and child.func.id == "expect") or (
          isinstance(child.func, ast.Attribute) and child.func.attr == "expect"
      ):
        asserts += 1
      if (
          isinstance(child.func, ast.Attribute)
          and child.func.attr == "locator"
          and child.args
          and isinstance(child.args[0], ast.Constant)
          and not child.args[0].value
      ):
        errors.append(f"Empty locator 'page.locator(\"\")' in '{node.name}'.")
  if not asserts:
    errors.append(f"Function '{node.name}' has 0 assertions or expects.")
  return errors


def lint_file(file_path: pathlib.Path) -> List[Tuple[str, int, str]]:
  """Lints a single python test file for vacuous assertions."""
  results = []
  try:
    tree = ast.parse(
        file_path.read_text(encoding="utf-8"), filename=str(file_path)
    )
    for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
        for err in check_vacuity(node):
          results.append((str(file_path), node.lineno, err))
  except SyntaxError as e:
    results.append((str(file_path), e.lineno or 0, f"SyntaxError: {e.msg}"))
  return results


def main() -> int:
  paths = [
      pathlib.Path(p) for p in sys.argv[1:] or ["third_party/xprof/tests/ui"]
  ]
  violations = []
  for p in paths:
    files = [p] if p.is_file() else p.rglob("test_*.py")
    for f in files:
      violations.extend(lint_file(f))

  if not violations:
    print(
        "Gate 1 Vacuity Linter: All test files passed cleanly (0 violations)."
    )
    return 0

  print(f"Gate 1 Vacuity Linter: Found {len(violations)} violation(s):")
  for path, line, msg in violations:
    print(f"  {path}:{line}: [VACUITY] {msg}")
  return 1


if __name__ == "__main__":
  sys.exit(main())
