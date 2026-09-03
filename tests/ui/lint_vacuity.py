"""Gate 1: Static AST Vacuity Linter for Playwright Test Suites."""

import ast
import os
import pathlib
import sys


def get_default_scan_dir() -> pathlib.Path:
  """Returns the directory containing UI test files, respecting runfiles."""
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


def check_vacuity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
  """Checks a single test function for anti-patterns."""
  errors = []
  asserts = 0
  for child in ast.walk(node):
    if isinstance(child, ast.If):
      for n in ast.walk(child):
        if isinstance(n, ast.Assert) or (
            isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Name, ast.Attribute))
            and (
                getattr(n.func, "id", None) == "expect"
                or getattr(n.func, "attr", None) == "expect"
                or (
                    isinstance(getattr(n.func, "attr", None), str)
                    and getattr(n.func, "attr", "").startswith("assert")
                )
            )
        ):
          errors.append(f"Conditional assertion detected in '{node.name}'.")
          break
    elif isinstance(child, ast.IfExp):
      for n in ast.walk(child):
        if isinstance(n, ast.Assert) or (
            isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Name, ast.Attribute))
            and (
                getattr(n.func, "id", None) == "expect"
                or getattr(n.func, "attr", None) == "expect"
                or (
                    isinstance(getattr(n.func, "attr", None), str)
                    and getattr(n.func, "attr", "").startswith("assert")
                )
            )
        ):
          errors.append(
              "Conditional assertion in ternary expression detected in"
              f" '{node.name}'."
          )
          break
    elif isinstance(child, ast.Assert):
      asserts += 1
    elif isinstance(child, ast.Call):
      if (
          isinstance(child.func, ast.Name)
          and (
              child.func.id == "expect"
              or child.func.id.startswith("assert")
              or child.func.id.startswith("_assert")
          )
      ) or (
          isinstance(child.func, ast.Attribute)
          and (
              child.func.attr == "expect"
              or child.func.attr.startswith("assert")
          )
      ):
        asserts += 1
      if isinstance(child.func, ast.Attribute) and child.func.attr == "locator":
        if (
            child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
            and not child.args[0].value.strip()
        ):
          errors.append(
              "Empty or whitespace locator"
              f" page.locator('{child.args[0].value}') in '{node.name}'."
          )
        for kw in child.keywords:
          if (
              kw.arg in ("selector", "selector_or_locator")
              and isinstance(kw.value, ast.Constant)
              and isinstance(kw.value.value, str)
              and not kw.value.value.strip()
          ):
            errors.append(
                "Empty or whitespace locator keyword"
                f" selector='{kw.value.value}' in '{node.name}'."
            )
  if not asserts and node.name.startswith("test"):
    errors.append(f"Function '{node.name}' has 0 assertions or expects.")
  return errors


def lint_file(file_path: pathlib.Path) -> list[tuple[str, int, str]]:
  """Lints a single python test file for vacuous assertions."""
  results = []
  try:
    tree = ast.parse(
        file_path.read_text(encoding="utf-8"), filename=str(file_path)
    )
    for node in ast.walk(tree):
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for err in check_vacuity(node):
          results.append((str(file_path), node.lineno, err))
  except SyntaxError as e:
    results.append((str(file_path), e.lineno or 0, f"SyntaxError: {e.msg}"))
  return results


def main() -> int:
  """Entry point for executing the static AST vacuity linter."""
  default_dir = get_default_scan_dir()
  paths = [pathlib.Path(p) for p in sys.argv[1:] or [default_dir]]
  scanned_files = 0
  violations = []
  for p in paths:
    files = [p] if p.is_file() else list(p.rglob("test_*.py"))
    for f in files:
      scanned_files += 1
      violations.extend(lint_file(f))

  if scanned_files == 0:
    print(
        "Gate 1 Vacuity Linter: FAILED (0 test files discovered; hermetic"
        " sandbox runfiles vacuity detected)."
    )
    return 1

  if not violations:
    print(
        f"Gate 1 Vacuity Linter: All {scanned_files} test file(s) passed"
        " cleanly (0 violations)."
    )
    return 0

  print(f"Gate 1 Vacuity Linter: Found {len(violations)} violation(s):")
  for path, line, msg in violations:
    print(f"  {path}:{line}: [VACUITY] {msg}")
  return 1


if __name__ == "__main__":
  sys.exit(main())
