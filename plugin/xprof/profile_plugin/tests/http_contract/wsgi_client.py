# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Minimal WSGI client for profile_plugin HTTP contract tests.

Always resolve apps from ``plugin.get_plugin_apps()`` so logging middleware and
``@wrappers.Request.application`` wiring are exercised.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Any, Mapping

from werkzeug.test import EnvironBuilder


@dataclass(frozen=True)
class WsgiResult:
  status: str
  status_code: int
  headers: list[tuple[str, str]]
  body: bytes


def _parse_status_code(status: str) -> int:
  return int(status.split(' ', 1)[0])


def call_route(
    plugin,
    route: str,
    *,
    method: str = 'GET',
    path: str | None = None,
    query: Mapping[str, str] | None = None,
) -> WsgiResult:
  """Invoke the registered WSGI app for ``route`` and return status/headers/body."""
  apps = plugin.get_plugin_apps()
  if route not in apps:
    raise KeyError(f'route not registered: {route!r}; known={sorted(apps)}')
  app = apps[route]
  env = EnvironBuilder(
      path=path if path is not None else route,
      method=method,
      query_string=dict(query or {}),
  ).get_environ()
  captured: dict[str, Any] = {}

  def start_response(status, headers, exc_info=None):
    captured['status'] = status
    captured['headers'] = list(headers)

  body = b''.join(app(env, start_response))
  status = captured.get('status', '500 Internal Server Error')
  return WsgiResult(
      status=status,
      status_code=_parse_status_code(status),
      headers=captured.get('headers', []),
      body=body,
  )


def header_value(result: WsgiResult, name: str) -> str | None:
  name_l = name.lower()
  for key, value in result.headers:
    if key.lower() == name_l:
      return value
  return None


def decode_body(result: WsgiResult) -> bytes:
  enc = header_value(result, 'Content-Encoding')
  if enc and 'gzip' in enc.lower():
    return gzip.decompress(result.body)
  return result.body


def json_body(result: WsgiResult) -> Any:
  return json.loads(decode_body(result))


def text_body(result: WsgiResult) -> str:
  return decode_body(result).decode('utf-8')
