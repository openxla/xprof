# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Frontend remote profile capture (/capture_profile)."""

from __future__ import annotations

from werkzeug import wrappers

from xprof.profile_plugin.http.respond import respond
from xprof.profile_plugin.lazy_imports import load_pywrap_module


class CaptureApiMixin:
  """HTTP handlers for this frontend API group (mixed into ProfilePlugin)."""

  @wrappers.Request.application
  def capture_route(self, request: wrappers.Request) -> wrappers.Response:
    # pytype: enable=wrong-arg-types
    return self.capture_route_impl(request)


  def capture_route_impl(self, request: wrappers.Request) -> wrappers.Response:
    """Runs the client trace for capturing profiling information."""
    service_addr = request.args.get('service_addr')
    duration = int(request.args.get('duration', '1000'))
    is_tpu_name = request.args.get('is_tpu_name') == 'true'
    worker_list = request.args.get('worker_list')
    num_tracing_attempts = int(request.args.get('num_retry', '0')) + 1
    options = {
        'host_tracer_level': int(request.args.get('host_tracer_level', '2')),
        'device_tracer_level': int(
            request.args.get('device_tracer_level', '1')
        ),
        'python_tracer_level': int(
            request.args.get('python_tracer_level', '0')
        ),
        'delay_ms': int(request.args.get('delay', '0')),
    }

    if is_tpu_name:
      if self._tf_profiler is None:
        return respond(
            {
                'error': (
                    'TensorFlow is not installed, but is required to use TPU'
                    ' names.'
                )
            },
            'application/json',
            code=500,
        )
      try:
        # Delegate to the helper class for all TF-related logic.
        service_addr, worker_list, master_ip = (
            self._tf_profiler.resolve_tpu_name(service_addr, worker_list or '')
        )
        self.master_tpu_unsecure_channel = master_ip
      except (RuntimeError, ValueError) as err:
        return respond({'error': str(err)}, 'application/json', code=500)

    if not self.logdir:
      return respond(
          {'error': 'logdir is not set, abort capturing.'},
          'application/json',
          code=500,
      )
    try:
      # The core trace call remains, now with cleanly resolved parameters.
      load_pywrap_module().trace(
          service_addr.removeprefix('grpc://'),
          str(self.logdir),
          worker_list,
          True,
          duration,
          num_tracing_attempts,
          options,
      )
      return respond(
          {'result': 'Capture profile successfully. Please refresh.'},
          'application/json',
      )
    except Exception as e:  # pylint: disable=broad-except
      return respond({'error': str(e)}, 'application/json', code=500)


