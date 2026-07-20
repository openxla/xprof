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
"""Optional TensorFlow helpers for remote profile capture."""

from __future__ import annotations

from xprof.profile_plugin.logging_config import logger

try:
  import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  tf.enable_v2_behavior()
except ImportError:
  logger.info(
      'Disabling some remote capture features as tensorflow is not available'
  )
  tf = None


class TfProfiler:
  """A helper class to encapsulate all TensorFlow-dependent profiler logic."""

  def __init__(self, tf_module):
    if not tf_module:
      raise ImportError('TensorFlow module is not available.')
    self.tf = tf_module

  def _get_worker_list(self, cluster_resolver) -> str:
    """Parses TPU workers list from the cluster resolver."""
    cluster_spec = cluster_resolver.cluster_spec()
    task_indices = cluster_spec.task_indices('worker')
    worker_list = [
        cluster_spec.task_address('worker', i).replace(':8470', ':8466')
        for i in task_indices
    ]
    return ','.join(worker_list)

  def resolve_tpu_name(
      self, tpu_name: str, worker_list: str
  ) -> tuple[str, str, str]:
    """Resolves a TPU name to its master IP, service address, and worker list.

    Args:
      tpu_name: The name of the TPU to resolve.
      worker_list: A comma-separated list of worker addresses.

    Returns:
      A tuple containing (service_addr, worker_list, master_ip).
    """
    try:
      resolver = self.tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu_name
      )
      master_grpc_addr = resolver.get_master()
    except RuntimeError as err:
      # Propagate error to be handled by the caller.
      raise RuntimeError(
          f'Error initializing TPUClusterResolver: {err}'
      ) from err
    except (ValueError, TypeError) as e:
      # Handle cases where the TPU name is invalid.
      raise ValueError(f'No TPU found with the name: {tpu_name}') from e

    if not worker_list:
      worker_list = self._get_worker_list(resolver)

    # TPU cluster resolver always returns port 8470. Replace it with 8466
    # on which profiler service is running.
    master_ip = master_grpc_addr.replace('grpc://', '').replace(':8470', '')
    service_addr = f'{master_ip}:8466'
    return service_addr, worker_list, master_ip


def create_tf_profiler() -> TfProfiler | None:
  """Returns a TfProfiler if TensorFlow is available, else None."""
  return TfProfiler(tf) if tf else None
