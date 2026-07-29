"""Unit tests for OSS xplane_tools.iter_planes.

These tests verify the fixes for the two review comments on cl/954632105:
1. Non-existent absolute paths raise FileNotFoundError instead of silently
   bypassing the XProf server fallback.
2. Directory globs search for both .xplane.pb and .xspace.pb files, matching
   the behavior of xprof_client.get_xspace_paths.
"""

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from xprof.cli.internal.oss import xplane_tools


class IterPlanesTest(absltest.TestCase):

  def test_nonexistent_absolute_path_raises_filenotfounderror(self):
    """A path starting with / that doesn't exist should raise FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      list(xplane_tools.iter_planes("/nonexistent/path/that/does/not/exist"))

  def test_empty_directory_raises_filenotfounderror(self):
    """An existing directory with no profile files should raise FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(FileNotFoundError):
        list(xplane_tools.iter_planes(tmpdir))

  def test_directory_glob_finds_both_xplane_and_xspace_files(self):
    """iter_planes should find both .xplane.pb and .xspace.pb files in a dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
      xplane_file = pathlib.Path(tmpdir) / "trace.xplane.pb"
      xspace_file = pathlib.Path(tmpdir) / "trace.xspace.pb"
      xplane_file.write_bytes(b"fake_xplane_data")
      xspace_file.write_bytes(b"fake_xspace_data")

      fake_pd = mock.MagicMock()
      fake_pd.planes = ["plane1", "plane2"]
      with mock.patch.object(
          xplane_tools.profiler.ProfileData,
          "from_serialized_xspace",
          return_value=fake_pd,
      ):
        planes = list(xplane_tools.iter_planes(tmpdir))

      # Both files should have been processed, yielding planes from each.
      self.assertLen(planes, 4)  # 2 planes * 2 files

  def test_existing_file_path_yields_planes(self):
    """An existing file path should be read and its planes yielded."""
    with tempfile.TemporaryDirectory() as tmpdir:
      trace_file = pathlib.Path(tmpdir) / "trace.xplane.pb"
      trace_file.write_bytes(b"fake_data")

      fake_pd = mock.MagicMock()
      fake_pd.planes = ["plane1"]
      with mock.patch.object(
          xplane_tools.profiler.ProfileData,
          "from_serialized_xspace",
          return_value=fake_pd,
      ):
        planes = list(xplane_tools.iter_planes(str(trace_file)))

      self.assertEqual(planes, ["plane1"])

  def test_nonexistent_relative_path_falls_through_to_server(self):
    """A relative session ID should fall through to the XProf server logic."""
    with mock.patch.object(
        xplane_tools.xprof_client, "get_client"
    ) as mock_get_client:
      mock_client = mock.MagicMock()
      mock_client.get_run_dir.return_value = pathlib.Path("/fake/run_dir")
      mock_client.get_xspace_paths.return_value = []
      mock_get_client.return_value = mock_client

      # Should not raise FileNotFoundError; should try server path.
      list(xplane_tools.iter_planes("nonexistent_relative_session_id"))
      mock_get_client.assert_called_once()


if __name__ == "__main__":
  absltest.main()
