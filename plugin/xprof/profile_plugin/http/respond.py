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
"""HTTP response helpers for the profile plugin."""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from typing import Any

from werkzeug import wrappers

from xprof import version


def respond(
    body: Any,
    content_type: str,
    code: int = 200,
    content_encoding: tuple[str, str] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> wrappers.Response:
  """Create a Werkzeug response, handling JSON serialization and CSP.

  Args:
    body: For JSON responses, a JSON-serializable object; otherwise, a raw
      `bytes` string or Unicode `str` (which will be encoded as UTF-8).
    content_type: Response content-type (`str`); use `application/json` to
      automatically serialize structures.
    code: HTTP status code (`int`).
    content_encoding: Response Content-Encoding header ('str'); e.g. 'gzip'. If
      the content type is not set, The data would be compressed and the content
      encoding would be set to gzip.
    extra_headers: Optional additional headers to add to the response.

  Returns:
    A `werkzeug.wrappers.Response` object.
  """
  if content_type == 'application/json' and isinstance(
      body, (dict, list, set, tuple)
  ):
    body = json.dumps(body, sort_keys=True)
  if not isinstance(body, bytes):
    body = body.encode('utf-8')
  csp_parts = {
      'default-src': ["'self'"],
      'script-src': [
          "'self'",
          "'unsafe-eval'",
          "'unsafe-inline'",
          'https://www.gstatic.com',
      ],
      'object-src': ["'none'"],
      'style-src': [
          "'self'",
          "'unsafe-inline'",
          'https://fonts.googleapis.com',
          'https://www.gstatic.com',
      ],
      'font-src': [
          "'self'",
          'https://fonts.googleapis.com',
          'https://fonts.gstatic.com',
          'data:',
      ],
      'connect-src': [
          "'self'",
          'data:',
          'www.gstatic.com',
      ],
      'img-src': [
          "'self'",
          'blob:',
          'data:',
      ],
      'frame-src': [
          "'self'",
          'https://ui.perfetto.dev',
      ],
      'script-src-elem': [
          "'self'",
          "'unsafe-inline'",
          # Remember to restrict on integrity when importing from jsdelivr
          # Whitelist this domain to support hlo_graph_dumper html format
          'https://cdn.jsdelivr.net/npm/',
          'https://www.gstatic.com',
      ],
  }
  csp = ';'.join((' '.join([k] + v) for (k, v) in csp_parts.items()))
  headers = [
      ('Content-Security-Policy', csp),
      ('X-Content-Type-Options', 'nosniff'),
  ]
  if content_encoding:
    headers.append(('Content-Encoding', content_encoding))
  else:
    headers.append(('Content-Encoding', 'gzip'))
    body = gzip.compress(body)

  if extra_headers is not None:
    headers.extend(extra_headers.items())

  return wrappers.Response(
      body, content_type=content_type, status=code, headers=headers
  )


@wrappers.Request.application
def version_route(_: wrappers.Request) -> wrappers.Response:
  return respond(version.__version__, 'text/plain')
