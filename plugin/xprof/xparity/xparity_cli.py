"""CLI for Xparity numerical accuracy and parity verification tools.

Exit codes (stable contract for autonomous agents):
  0  success
  1  INTERNAL_ERROR   unexpected failure inside the validator
  2  USAGE_ERROR      malformed invocation, unknown subcommand or flag
  3  PATH_ERROR       a suite or output path could not be read or written
  4  INVALID_VALUE    a supplied value is outside its permitted domain
"""

import sys
from typing import Any

from absl import app
from absl import flags
import fire

from xprof.xparity import xparity_tool


EXIT_OK = 0
EXIT_INTERNAL_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_PATH_ERROR = 3
EXIT_INVALID_VALUE = 4

_BUG_LINK = "https://github.com/openxla/xprof/issues"


def cli_main() -> dict[str, Any]:
  """Initializes the Xparity CLI and returns the available subcommands."""
  return {
      # keep-sorted start
      "generate_suite": xparity_tool.generate_suite,
      "inspect_suite": xparity_tool.inspect_suite,
      "probe_precision": xparity_tool.probe_precision,
      "verify": xparity_tool.verify_numerical_parity,
      "verify_numerical_parity": xparity_tool.verify_numerical_parity,
      # keep-sorted end
  }


def _classify(err: BaseException) -> tuple[int, str]:
  """Maps an exception to a stable (exit_code, taxonomy_label) pair."""
  if isinstance(err, (FileNotFoundError, IsADirectoryError, PermissionError)):
    return EXIT_PATH_ERROR, "PATH_ERROR"
  if isinstance(err, (KeyError, ValueError)):
    return EXIT_INVALID_VALUE, "INVALID_VALUE"
  if isinstance(err, TypeError):
    return EXIT_USAGE_ERROR, "USAGE_ERROR"
  return EXIT_INTERNAL_ERROR, "INTERNAL_ERROR"


def main(argv: list[str] | None = None) -> int:
  """Dispatches a subcommand and returns a process exit code."""
  if argv is None:
    argv = sys.argv
  try:
    fire.Fire(cli_main(), command=argv[1:] if len(argv) > 1 else None)
  except SystemExit as err:  # Raised by Fire on a usage error.
    code = err.code if isinstance(err.code, int) else EXIT_USAGE_ERROR
    return EXIT_OK if code == EXIT_OK else EXIT_USAGE_ERROR
  except Exception as err:  # pylint: disable=broad-except
    code, label = _classify(err)
    print(
        f"[xparity:{label}] {type(err).__name__}: {err}\n"
        f"If this looks like a tool defect, please file at {_BUG_LINK}.",
        file=sys.stderr,
    )
    return code
  return EXIT_OK


def _absl_main(argv: list[str]) -> None:
  """absl entry point; translates the return code into a process exit."""
  code = main(argv)
  if code != EXIT_OK:
    sys.exit(code)


if __name__ == "__main__":
  # `known_only=True` lets absl consume only the flags it defines and leaves
  # Fire-style `--key=value` subcommand arguments in argv. Without it absl
  # aborts with "Unknown command line flag" before main() is ever entered,
  # which makes every documented `--flag` invocation unusable.
  app.run(
      _absl_main,
      flags_parser=lambda argv: flags.FLAGS(argv, known_only=True),
  )
