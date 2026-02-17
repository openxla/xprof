"""Integration test for JAX profiling on TPU."""

import glob
import os

from absl import flags
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp


FLAGS = flags.FLAGS


def run_jax_workload():
  """Runs a simple JAX matrix multiplication workload on TPU.

  This function creates random matrices, JIT compiles a matmul function,
  runs a warmup, and then executes the workload multiple times with
  step trace annotations for profiling.
  """
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (1024, 1024))
  y = jax.random.normal(key, (1024, 1024))

  @jax.jit
  def matmul(a, b):
    return jnp.matmul(a, b)

  # Warmup
  matmul(x, y).block_until_ready()

  # Run with XProf Profiler.
  for step in range(5):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      matmul(x, y).block_until_ready()


class JaxProfilerTest(absltest.TestCase):

  def test_profile_jax_workload(self):
    """Verifies that JAX profiling captures traces on TPU."""
    logdir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not logdir:
      # Fallback for local run if env var not set
      logdir = self.create_tempdir().full_path

    logging.info("Profiling to logdir: %s", logdir)

    options = jax.profiler.ProfileOptions()
    logging.info("Starting JAX workload with profiling...")

    with jax.profiler.trace(logdir, profiler_options=options):
      run_jax_workload()

    # Verify that .xplane.pb files are generated.
    path = os.path.join(logdir, "plugins", "profile", "*", "*.xplane.pb")
    found_files = glob.glob(path)

    logging.info("Checking for .xplane.pb files at: %s", path)
    logging.info("Found files: %s", found_files)

    self.assertNotEmpty(found_files, f"Expected .xplane.pb files in {path}")

    for f in found_files:
      size = os.path.getsize(f)
      logging.info("File: %s, size: %d", f, size)
      self.assertGreater(size, 0, f"File {f} is empty")


if __name__ == "__main__":
  jax.config.config_with_absl()
  absltest.main()
