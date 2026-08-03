# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Request argument parsing helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from werkzeug import wrappers


def get_bool_arg(args: Mapping[str, Any], arg_name: str, default: bool) -> bool:
  """Gets a boolean argument from a request.

  Args:
    args: The werkzeug request arguments.
    arg_name: The name of the argument.
    default: The default value if the argument is not present.

  Returns:
    The boolean value of the argument.
  """
  arg_str = args.get(arg_name)
  if arg_str is None:
    return default
  return arg_str.lower() == 'true'


def generate_csv_filename(request: wrappers.Request) -> str:
  """Generates a sanitized filename for the CSV export."""
  tool = request.args.get('tag', 'data')
  run = request.args.get('run', '')
  host = request.args.get('host', '')

  safe_tool = re.sub(r'[^a-zA-Z0-9_\-]', '_', tool)
  safe_run = re.sub(r'[^a-zA-Z0-9_\-]', '_', run)

  host_suffix = host.split('-')[-1] if host else ''
  safe_host_suffix = re.sub(r'[^a-zA-Z0-9_\-]', '_', host_suffix)

  filename_parts = (p for p in [safe_tool, safe_run, safe_host_suffix] if p)
  return '_'.join(filename_parts) + '.csv'
