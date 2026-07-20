# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend background cache warm-up (/generate_cache)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from werkzeug import wrappers

from xprof.profile_plugin.http.respond import respond
from xprof.profile_plugin.logging_config import logger
from xprof.profile_plugin.tools.filenames import hosts_from_xplane_filenames
from xprof.profile_plugin.tools.registry import DEFAULT_CACHE_TOOLS, XPLANE_TOOLS_SET


class CacheApiMixin:
  """HTTP handlers for this frontend API group (mixed into ProfilePlugin)."""

  @wrappers.Request.application
  def generate_cache_route(
      self, request: wrappers.Request
  ) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    """Generates tool data cache in the background."""
    return self._generate_cache_impl(request)


  def _generate_cache_impl(
      self, request: wrappers.Request
  ) -> wrappers.Response:
    """Generates tool data cache in the background.

    Args:
      request: The Werkzeug request object. `request.args` may contain the
        following parameters: - session_path: The path to the session directory
        containing XPlane files. (Required) - tools: An optional comma-separated
        list of tool names to generate cache for. If not provided, defaults to
        `DEFAULT_CACHE_TOOLS`.

    Returns:
      A JSON response indicating whether the task was accepted.
    """
    logger.info('Received generate_cache request.')
    if request.method != 'POST':
      return respond('Method Not Allowed', 'text/plain', code=405)

    params = request.args
    session_path = params.get('session_path')
    logger.info('generate_cache called with params: %s', params)

    if not session_path:
      return respond('Missing "session_path" parameter', 'text/plain', code=400)

    try:
      path = self._epath.Path(session_path)
      xplane_basenames = self._get_xplane_basenames(session_path)
      asset_paths = sorted(
          str(path / basename) for basename in xplane_basenames
      )
      if not asset_paths:
        return respond(
            'No XPlane files found in session_path', 'text/plain', code=404
        )
      logger.info(
          'Found %d *.xplane.pb files in %s.', len(asset_paths), session_path
      )
    except OSError as e:
      logger.exception('Error listing files in session_path: %s', session_path)
      return respond(
          f'Error listing files in session_path: {e!r}', 'text/plain', code=500
      )

    runs = self.runs_imp(request)
    if len(runs) != 1:
      # When 'session_path' is provided, runs_imp should return exactly one run
      # corresponding to the session_path's name.
      return respond(
          'Expected exactly one run for the provided session_path, but found'
          ' %d: %s. Please ensure session_path points to a valid profile run'
          ' directory.' % (len(runs), runs),
          'text/plain',
          code=400,
      )
    run_name = runs[0]
    logger.info(
        'Querying available tools for run %s via run_tools_imp.',
        run_name,
    )
    available_run_tools = set(self.run_tools_imp(run_name, request))
    logger.info(
        'Discovered tools for cache generation: %s for run %s',
        available_run_tools,
        run_name,
    )

    tools_str = params.get('tools')
    requested_tools = (
        set(t.strip() for t in tools_str.split(',') if t.strip())
        if tools_str
        else set(DEFAULT_CACHE_TOOLS)
    )
    if tools_str:
      logger.info('Request tools for cache generation: %s', requested_tools)
    else:
      logger.info(
          'No tools specified in request, using default tools: %s',
          DEFAULT_CACHE_TOOLS,
      )

    available_xplane_tools = available_run_tools.intersection(XPLANE_TOOLS_SET)

    filtered_tools = requested_tools.intersection(available_xplane_tools)

    skipped_tools = requested_tools.difference(filtered_tools)
    for tool in skipped_tools:
      if tool not in available_run_tools:
        logger.info(
            'Tool %s was requested for caching but is not available for run %s,'
            ' skipping.',
            tool,
            run_name,
        )
      else:
        logger.warning(
            'Tool %s is available for run %s but not in XPLANE_TOOLS_SET,'
            ' skipping cache generation.',
            tool,
            run_name,
        )

    if not filtered_tools:
      return respond(
          'No valid XPlane tools found or specified for caching in run %s.'
          % run_name,
          'text/plain',
          code=400,
      )

    logger.info(
        'Filtered tools for cache generation: %s for session %s',
        filtered_tools,
        session_path,
    )

    try:
      logger.info(
          'Submitting cache generation task to thread pool for session %s...',
          session_path,
      )
      self._cache_generation_pool.submit(
          self._generate_cache_task,
          asset_paths=asset_paths,
          tool_list=sorted(list(filtered_tools)),
          params=params,
          session_path=session_path,
      )
    except RuntimeError as e:
      logger.exception(
          'Failed to schedule cache generation task for session_path: %s',
          session_path,
      )
      return respond(f'Failed to schedule task: {e!r}', 'text/plain', code=500)
    else:
      return respond(
          {'status': 'ACCEPTED', 'message': 'Cache generation started'},
          'application/json',
          code=202,
      )


  def _generate_cache_task(
      self,
      *,
      asset_paths: Sequence[str],
      tool_list: Sequence[str],
      params: Mapping[str, Any],
      session_path: str,
  ) -> None:
    """Generates and caches tool data from XPlane files in a background thread.

    Args:
      asset_paths: A list of paths to the XPlane files.
      tool_list: A list of tool names for which to generate cache.
      params: Additional parameters from the request.
      session_path: The path to the session directory.
    """
    logger.info(
        'Background cache generation task started for tools: %s', tool_list
    )
    logger.info('Writing cache version file to %s', session_path)
    self._write_cache_version_file(session_path)

    filenames = [os.path.basename(p) for p in asset_paths]

    base_tool_params = dict(params)

    for tool in tool_list:
      try:
        logger.info('Generating cache for tool %s...', tool)
        tool_params = base_tool_params.copy()
        tool_params['hosts'] = hosts_from_xplane_filenames(filenames, tool)
        self._get_xspace_fn()(
            [self._epath.Path(p) for p in asset_paths], tool, tool_params
        )
        logger.info(
            'Successfully generated cache for tool %s for %d files.',
            tool,
            len(asset_paths),
        )
      # Catch all exceptions to prevent the background thread from crashing.
      # This ensures that even if one tool fails to generate, other tools
      # can still be processed. The error is logged for debugging.
      except (AttributeError, ValueError, OSError):
        logger.exception(
            'Background cache generation failed for tool %s in session %s',
            tool,
            session_path,
        )
      except Exception:  # pylint: disable=broad-except
        logger.exception(
            'Unexpected error during background cache generation for tool %s'
            ' in session %s',
            tool,
            session_path,
        )


