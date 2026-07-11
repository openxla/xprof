"""Tests for optional TensorFlow TPU resolution helpers."""

from __future__ import annotations

import types
import unittest
from unittest import mock

from xprof.profile_plugin.tensorflow_bridge import TfProfiler, create_tf_profiler


class _FakeClusterSpec:
  def task_indices(self, job):
    return [0, 1]

  def task_address(self, job, i):
    return f'worker{i}:8470'


class _FakeResolver:
  def __init__(self, tpu_name):
    self.tpu_name = tpu_name

  def cluster_spec(self):
    return _FakeClusterSpec()

  def get_master(self):
    return 'grpc://10.0.0.1:8470'


class TensorflowBridgeTest(unittest.TestCase):

  def test_create_tf_profiler_none_without_tf(self):
    with mock.patch('xprof.profile_plugin.tensorflow_bridge.tf', None):
      self.assertIsNone(create_tf_profiler())

  def test_tf_profiler_requires_module(self):
    with self.assertRaises(ImportError):
      TfProfiler(None)

  def test_resolve_tpu_name_builds_service_addr_and_workers(self):
    tf_mod = types.SimpleNamespace(
        distribute=types.SimpleNamespace(
            cluster_resolver=types.SimpleNamespace(
                TPUClusterResolver=_FakeResolver
            )
        )
    )
    profiler = TfProfiler(tf_mod)
    service_addr, workers, master_ip = profiler.resolve_tpu_name('my-tpu', '')
    self.assertEqual(master_ip, '10.0.0.1')
    self.assertEqual(service_addr, '10.0.0.1:8466')
    self.assertEqual(workers, 'worker0:8466,worker1:8466')

  def test_resolve_tpu_name_uses_provided_worker_list(self):
    tf_mod = types.SimpleNamespace(
        distribute=types.SimpleNamespace(
            cluster_resolver=types.SimpleNamespace(
                TPUClusterResolver=_FakeResolver
            )
        )
    )
    profiler = TfProfiler(tf_mod)
    service_addr, workers, master_ip = profiler.resolve_tpu_name(
        'my-tpu', 'a:8466,b:8466'
    )
    self.assertEqual(workers, 'a:8466,b:8466')
    self.assertEqual(service_addr, '10.0.0.1:8466')

  def test_resolve_tpu_name_invalid_raises_value_error(self):
    def boom(tpu_name):
      raise ValueError('bad')

    tf_mod = types.SimpleNamespace(
        distribute=types.SimpleNamespace(
            cluster_resolver=types.SimpleNamespace(TPUClusterResolver=boom)
        )
    )
    profiler = TfProfiler(tf_mod)
    with self.assertRaises(ValueError):
      profiler.resolve_tpu_name('missing', '')


if __name__ == '__main__':
  unittest.main()
