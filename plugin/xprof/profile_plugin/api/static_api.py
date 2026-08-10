# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend static assets + UI config (index, JS/CSS/WASM, /config)."""

from __future__ import annotations

import os
from pathlib import Path

from werkzeug import wrappers

from xprof.profile_plugin.http.respond import respond
from xprof.profile_plugin.logging_config import logger
from xprof.standalone.tensorboard_shim import base_plugin

# xprof/ package root (contains static/); this file is profile_plugin/api/*.py
_STATIC_DIR = Path(__file__).resolve().parents[2] / 'static'


class StaticApiMixin:
  """HTTP handlers for this frontend API group (mixed into ProfilePlugin)."""

  @wrappers.Request.application
  def default_handler(self, _: wrappers.Request) -> wrappers.Response:
    contents = self._read_static_file_impl('index.html')
    return respond(contents, 'text/html')


  @wrappers.Request.application
  def config_route(self, _: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    """Returns UI configuration details."""
    config_data = {
        'hideCaptureProfileButton': self.hide_capture_profile_button,
        'srcPathPrefix': self.src_prefix,
        'enableTabNameLabel': self.enable_tab_name_label,
    }
    logger.info('config_route: %s', config_data)
    return respond(config_data, 'application/json')


  def frontend_metadata(self):
    return base_plugin.FrontendMetadata(es_module_path='/index.js')


  def _read_static_file_impl(self, filename: str) -> bytes:
    """Reads contents from a filename.

    Args:
      filename (str): Name of the file.

    Returns:
      Contents of the file.
    Raises:
      IOError: File could not be read or found.
    """
    filepath = str(_STATIC_DIR / filename)

    try:
      with open(filepath, 'rb') as infile:
        contents = infile.read()
    except IOError as io_error:
      raise io_error
    return contents


  @wrappers.Request.application
  def static_file_route(self, request: wrappers.Request) -> wrappers.Response:
    """Handles static files."""
    # pytype: enable=wrong-arg-types
    filename = os.path.basename(request.path)
    extension = os.path.splitext(filename)[1]
    if extension == '.html':
      mimetype = 'text/html'
    elif extension == '.css':
      mimetype = 'text/css'
    elif extension == '.js':
      mimetype = 'application/javascript'
    elif extension == '.wasm':
      mimetype = 'application/wasm'
    else:
      mimetype = 'application/octet-stream'
    try:
      contents = self._read_static_file_impl(filename)
    except IOError:
      return respond('Fail to read the files.', 'text/plain', code=404)
    return respond(contents, mimetype)


