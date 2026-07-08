"""Internal utilities for fetching OSS XProf Smart Suggestions."""

from __future__ import annotations

import html.parser
import json
import logging
from typing import Any

from . import xprof_client


class _HTMLTagStripper(html.parser.HTMLParser):
  """An HTML parser for stripping HTML tags from text content."""

  def __init__(self):
    super().__init__()
    self.convert_charrefs = True
    self.text = []

  def handle_data(self, data):
    self.text.append(data)

  def get_data(self):
    return ''.join(self.text)


def _strip_html_recursive(obj: Any) -> Any:
  if isinstance(obj, dict):
    return {k: _strip_html_recursive(v) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [_strip_html_recursive(item) for item in obj]
  elif isinstance(obj, str):
    stripper = _HTMLTagStripper()
    stripper.feed(obj)
    stripper.close()
    return stripper.get_data()
  return obj


def fetch_smart_suggestions(
    session_id: str,
    *,
    strip_html: bool = False,
) -> dict[str, Any]:
  """Fetches Smart Suggestions for an XProf session.

  Args:
    session_id: The XProf session ID.
    strip_html: Whether to recursively strip HTML tags from the response.

  Returns:
    The suggestions dictionary, or a dict with 'error' key on failure.
  """
  try:
    client = xprof_client.get_client()
    _, body = client.fetch(
        tool_name='smart_suggestion.json',
        session_id=session_id,
    )

    if not body:
      return {
          'error': f'No smart suggestions returned for session {session_id}'
      }

    body = (
        body.decode('utf-8', errors='replace')
        if isinstance(body, bytes)
        else body
    )

    res_dict = json.loads(body)
    if strip_html:
      res_dict = _strip_html_recursive(res_dict)
    return res_dict

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception('Error fetching smart suggestions')
    return {'error': f'Error fetching smart suggestions: {e!r}'}
