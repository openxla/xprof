import glob
import json
import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp

from xprof.convert import raw_to_tool_data

FLAGS = flags.FLAGS


class PerfCountersTest(absltest.TestCase):

  def test_perf_counters_collected(self):
    k1 = jax.random.PRNGKey(0)
    k2 = jax.random.PRNGKey(1)
    logdir = FLAGS.test_tmpdir

    with jax.profiler.trace(logdir):
      # Generate random matrices
      x = jax.random.normal(k1, (128, 128))
      y = jax.random.normal(k2, (128, 128))

      # Matmul
      z = jnp.dot(x, y)

      # Add
      w = z + 1.0

      # Check shape and that it ran without error
      self.assertEqual(w.shape, (128, 128))

    profile_plugin_root = os.path.join(logdir, 'plugins/profile')
    # The session exists under a director whose name is time-dependent.
    profile_session_glob = os.path.join(profile_plugin_root, '*', '*.xplane.pb')
    xplane_files = glob.glob(profile_session_glob)
    self.assertLen(xplane_files, 1)

    perf_counters_data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xplane_files, 'perf_counters', {}
    )
    self.assertEqual(content_type, 'application/json')

    parsed_data = json.loads(perf_counters_data)
    self.assertIn('cols', parsed_data)
    self.assertIn('rows', parsed_data)


if __name__ == '__main__':
  absltest.main()
