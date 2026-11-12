# Copyright 2026 The XProf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
_NUM_BLOCKS = flags.DEFINE_integer(
    "num_blocks",
    2,
    "Timed blocks per role when more than one role runs. The blocks of the"
    " roles alternate, so a drift of the device over time does not favor one"
    " role.",
)


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
          num_blocks=_NUM_BLOCKS.value,
      )
  )


if __name__ == "__main__":
  app.run(main)
