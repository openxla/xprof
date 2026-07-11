# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend tool data (/data, /data_csv) and HLO module list."""

from __future__ import annotations

import os

from werkzeug import wrappers

from xprof.profile_plugin.cache.result_cache_policy import write_cache_version_file
from xprof.profile_plugin.http.parse_request import tool_request_from_args
from xprof.profile_plugin.http.request_params import generate_csv_filename
from xprof.profile_plugin.http.respond import respond
from xprof.profile_plugin.lazy_imports import load_convert_module
from xprof.profile_plugin.logging_config import logger
from xprof.profile_plugin.tools.filenames import parse_filename


class DataApiMixin:
  """HTTP handlers for this frontend API group (mixed into ProfilePlugin)."""

  @wrappers.Request.application
  def hlo_module_list_route(
      self, request: wrappers.Request
  ) -> wrappers.Response:
    module_names_str = self.hlo_module_list_impl(request)
    return respond(module_names_str, 'text/plain')


  def _write_cache_version_file(self, run_dir: str) -> None:
    """Persist the plugin version next to a session for result-cache invalidation.

    Thin delegate so tests can still patch this method; ToolDataService writes
    via ``result_cache_policy.write_cache_version_file`` directly.
    """
    write_cache_version_file(run_dir, self._version, self._epath)


  def data_impl(
      self, request: wrappers.Request
  ) -> tuple[bytes | str | None, str, str | None]:
    """Retrieves and processes the tool data for a run and a host.

    Args:
      request: XMLHttpRequest

    Returns:
      A string that can be served to the frontend tool or None if tool,
        run or host is invalid.

    Raises:
      FileNotFoundError: If a required xplane file for the specified host(s)
        is not found.
      IOError: If there is an error reading asset directories.
      AttributeError: If there is an error during xplane to tool data conversion
      ValueError: If xplane conversion fails due to invalid data.
      RuntimeError: If session resolution fails when logdir is missing.
    """
    req = tool_request_from_args(request.args)
    result = self._tool_data.get_tool_data(
        req,
        session_path=request.args.get('session_path'),
        run_path=request.args.get('run_path'),
        logdir=self.logdir,
        run_dir_cache=self._run_to_profile_run_dir,
        cache_lock=self._run_dir_cache_lock,
    )
    return result.data, result.content_type, result.content_encoding


  def hlo_module_list_impl(self, request: wrappers.Request) -> str:
    """Returns a string of HLO module names concatenated by comma for the given run."""
    run = request.args.get('run')
    run_dir = self._run_dir(run, request)
    if not run_dir:
      logger.warning('Cannot find asset directory for: %s', run)
      return ''
    try:
      all_basenames = self._profile_fs(run_dir).get_all_basenames(run_dir)
      module_list = [
          name
          for f in all_basenames
          if f.endswith('.hlo_proto.pb') and (name := parse_filename(f)[0])
      ]

      if not module_list:
        xplane_basenames = self._get_xplane_basenames(run_dir)
        xplane_filenames = [os.path.join(run_dir, f) for f in xplane_basenames]
        if xplane_filenames:
          try:
            # This triggers ConvertMultiXSpaceToHloProto in the C++ backend.
            load_convert_module().xspace_to_tool_names(xplane_filenames)
          except AttributeError:
            logger.warning(
                'XPlane converters are available after Tensorflow 2.4'
            )

          all_basenames = self._profile_fs(run_dir).get_all_basenames(run_dir)
          module_list = [
              name
              for f in all_basenames
              if f.endswith('.hlo_proto.pb') and (name := parse_filename(f)[0])
          ]

      return ','.join(module_list)
    except OSError as e:
      logger.warning('Cannot read asset directory: %s, OpError %r', run_dir, e)
      return ''


  @wrappers.Request.application
  def data_route(self, request: wrappers.Request) -> wrappers.Response:
    """Handlers for data."""
    # pytype: enable=wrong-arg-types
    # params
    #   request: XMLHTTPRequest.
    try:
      data, content_type, content_encoding = self.data_impl(request)
      if data is None:
        return respond('No Data', 'text/plain', code=404)
      return respond(data, content_type, content_encoding=content_encoding)
    # Data fetch error handler
    except TimeoutError as e:
      return respond(str(e), 'text/plain', code=500)
    except AttributeError as e:
      return respond(str(e), 'text/plain', code=500)
    except ValueError as e:
      return respond(str(e), 'text/plain', code=500)
    except FileNotFoundError as e:
      return respond(str(e), 'text/plain', code=500)
    except IOError as e:
      return respond(str(e), 'text/plain', code=500)


  @wrappers.Request.application
  # pytype: enable=wrong-arg-types
  def data_csv_route(self, request: wrappers.Request) -> wrappers.Response:
    """Retrieves tool data and converts it to CSV before responding."""
    try:
      data, content_type, _ = self.data_impl(request)

      if data is None:
        return respond('No Data Found', 'text/plain', code=404)

      if content_type != 'application/json':
        return respond(
            'CSV format not supported for this tool type',
            'text/plain',
            code=400,
        )

      csv_data = load_convert_module().json_to_csv_string(data)
      filename = generate_csv_filename(request)

      return respond(
          csv_data,
          'text/csv',
          content_encoding=None,
          extra_headers={
              'Content-Disposition': f'attachment; filename="{filename}"'
          },
      )

    except (
        TimeoutError,
        AttributeError,
        ValueError,
        FileNotFoundError,
        IOError,
        TypeError,
    ) as e:
      logger.exception('CSV conversion error')
      return respond(str(e), 'text/plain', code=500)


