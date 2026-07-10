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
"""WSGI logging middleware for profile plugin routes."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TextIO

from xprof.profile_plugin.logging_config import logger


def logging_wrapper(
    app: Callable[
        [
            Mapping[str, Any],
            Callable[[str, Sequence[tuple[str, str]], Any | None], Any],
        ],
        Iterable[bytes],
    ],
    clock: Callable[[], float] = time.time,
    writer: TextIO = sys.stderr,
) -> Callable[
    [
        Mapping[str, Any],
        Callable[[str, Sequence[tuple[str, str]], Any | None], Any],
    ],
    Iterable[bytes],
]:
  """Wraps a WSGI application to log request timing in GKE-friendly JSON format.

  Args:
    app: The WSGI application to wrap.
    clock: A function that returns the current time. Defaults to `time.time`.
    writer: A file-like object to write logs to. Defaults to `sys.stderr`.

  Returns:
    A new WSGI application that logs request timing.
  """

  def wrapper(
      environ: Mapping[str, Any],
      start_response: Callable[
          [str, Sequence[tuple[str, str]], Any | None], Any
      ],
  ) -> Iterable[bytes]:
    start_time = clock()
    request_method = environ.get('REQUEST_METHOD', '')
    # Reconstruct the URL as best as possible from WSGI environ.
    url_path = environ.get('SCRIPT_NAME', '') + environ.get('PATH_INFO', '')
    query_string = environ.get('QUERY_STRING')
    request_url = f'{url_path}?{query_string}' if query_string else url_path

    # container for status code captured from start_response
    status_info = {'code': 200}  # Default to 200 if not captured (unlikely)

    def custom_start_response(
        status: str,
        headers: Sequence[tuple[str, str]],
        exc_info: Any | None = None,
    ) -> Any:
      try:
        status_code_str, *_ = status.split(' ', 1)
        status_info['code'] = int(status_code_str)
      except (ValueError, IndexError):
        logger.warning(
            'Failed to parse status code from %r', status, exc_info=True
        )
      return start_response(status, headers, exc_info)

    response = app(environ, custom_start_response)
    try:
      yield from response
    finally:
      duration = clock() - start_time
      log_entry = {
          'httpRequest': {
              'requestMethod': request_method,
              'requestUrl': request_url,
              'status': status_info['code'],
              'latency': f'{duration:.9f}s',
          }
      }
      # Write raw JSON to stderr for GKE/Cloud Logging to pick up as structured
      # payload. We use sys.stderr instead of the logging module because the
      # latter would add text prefixes (timestamp, level, etc.) that break the
      # JSON parsing required for the "httpRequest" field to be recognized by
      # GKE.
      writer.write(json.dumps(log_entry) + '\n')
      writer.flush()
      if hasattr(response, 'close'):
        response.close()

  return wrapper
