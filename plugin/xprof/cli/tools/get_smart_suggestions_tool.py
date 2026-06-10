"""Tool for fetching Smart Suggestions for an XProf session."""

from __future__ import annotations

import json

from xprof.cli.internal import decorators
from xprof.cli.internal.oss import smart_suggestion_tools


@decorators.cached(expire=86400)
def get_smart_suggestions(session_id: str) -> str:
  """Fetches optimization suggestions for a given XProf session.

  These suggestions identify performance bottlenecks (e.g., input-bound,
  compute-bound) and provide actionable optimization recommendations.

  Args:
    session_id: The unique XProf session ID.

  Returns:
    A JSON-formatted string containing the list of suggestions and their rule
    names. On failure, returns a JSON string with 'error' and 'traceback' keys.
  """
  # Now uses the high-reliability HTTP-over-RPC logic exclusively.
  res = smart_suggestion_tools.fetch_smart_suggestions(
      session_id, strip_html=True
  )
  return json.dumps(res, indent=2)
