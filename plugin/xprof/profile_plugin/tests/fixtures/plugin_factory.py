# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Shared ProfilePlugin factory for characterization tests."""

from __future__ import annotations

import concurrent.futures
import json
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from etils import epath

from xprof.profile_plugin.plugin import ProfilePlugin
from xprof.standalone.tensorboard_shim import base_plugin
from xprof.standalone.tensorboard_shim import data_provider
from xprof.standalone.tensorboard_shim import plugin_event_multiplexer


class FakeFlags:
  def __init__(self, logdir, master_tpu_unsecure_channel=''):
    self.logdir = logdir
    self.master_tpu_unsecure_channel = master_tpu_unsecure_channel


def make_plugin(
    logdir: str,
    xspace_fn: Callable[
        [Sequence[Any], str, dict[str, Any]],
        tuple[Any, str],
    ]
    | None = None,
    version: str = '9.9.9',
) -> ProfilePlugin:
  """Construct ProfilePlugin with injectable convert and fixed version."""
  multiplexer = plugin_event_multiplexer.EventMultiplexer()
  multiplexer.AddRunsFromDirectory(logdir)
  context = base_plugin.TBContext(
      logdir=logdir,
      multiplexer=multiplexer,
      data_provider=data_provider.MultiplexerDataProvider(multiplexer, logdir),
      flags=FakeFlags(logdir),
  )
  default_fn = lambda paths, tool, params: (
      json.dumps({'ok': True, 'tool': tool}, sort_keys=True),
      'application/json',
  )
  return ProfilePlugin(
      context,
      epath_module=epath,
      xspace_to_tool_data_fn=xspace_fn or default_fn,
      version_module=SimpleNamespace(__version__=version),
      cache_generation_executor=concurrent.futures.ThreadPoolExecutor(
          max_workers=1
      ),
  )
