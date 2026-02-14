import glob
import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp

from xprof.convert import raw_to_tool_data

FLAGS = flags.FLAGS


class HloMetadataTest(absltest.TestCase):

  def test_hlo_metadata_collected(self):
    print('Devices:', jax.devices())
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

    result = raw_to_tool_data.xspace_to_tool_names(xplane_files)
    result.sort()
    expected = [
        'memory_viewer',
        'graph_viewer',
    ]
    expected.sort()
    self.assertContainsSubset(expected, result)

    print('Test finished successfully')


if __name__ == '__main__':
  absltest.main()
