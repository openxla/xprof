"""Command-line entry point for an XLoop worker process.

Kept separate from `xloop_worker_lib` on purpose. Flag definitions are global
process state, and required flags defined at import time would be inherited by
every binary that transitively imports the library -- including the `xprof`
CLI, which would then refuse to start without `--item_path`.
"""

from __future__ import annotations

import collections.abc
import sys

from absl import app
from absl import flags

from xprof.xloop import input_utils
from xprof.xloop import xloop_worker_lib

_Sequence = collections.abc.Sequence

_ITEM_PATH = flags.DEFINE_string(
    "item_path", None, "Path to the XLoopItem to execute.", required=True
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Directory for traces and results.", required=True
)
_CONFIG_NAME = flags.DEFINE_string(
    "config_name", "", "Input-generation config to run. Empty means the first."
)
_ROLES = flags.DEFINE_list(
    "roles", list(input_utils.ROLES), "Implementation roles to execute."
)
_NUM_WARMUPS = flags.DEFINE_integer("num_warmups", 3, "Untimed iterations.")
_NUM_EXECUTIONS = flags.DEFINE_integer(
    "num_executions", 10, "Timed iterations."
)
_SEED = flags.DEFINE_integer("seed", 0, "Input-generation seed.")


def main(argv: _Sequence[str]) -> None:
  del argv
  sys.exit(
      xloop_worker_lib.run_worker_reporting_failures(
          item_path=_ITEM_PATH.value,
          output_dir=_OUTPUT_DIR.value,
          config_name=_CONFIG_NAME.value,
          roles=_ROLES.value,
          num_warmups=_NUM_WARMUPS.value,
          num_executions=_NUM_EXECUTIONS.value,
          seed=_SEED.value,
      )
  )


if __name__ == "__main__":
  app.run(main)
