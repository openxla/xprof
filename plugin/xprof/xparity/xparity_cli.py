"""CLI for Xparity numerical accuracy and parity verification tools."""

import sys
from typing import Any

from absl import app
import fire

from xprof.xparity import xparity_tool


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


def main(argv: list[str] | None = None) -> None:
  if argv is None:
    argv = sys.argv
  fire.Fire(cli_main(), command=argv[1:] if len(argv) > 1 else None)


if __name__ == "__main__":
  app.run(main)
