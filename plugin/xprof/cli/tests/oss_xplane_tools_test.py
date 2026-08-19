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
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xplane_tools


class IterPlanesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache",
            return_value=mock_cache,
            autospec=True,
        )
    )

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

  def test_list_xplane_events_max_events_negative_one_returns_all(self):
    """Verifies max_events=-1 or 0 is treated as unlimited."""

    class _FakeEvent:

      def __init__(self, name, start_ns=0, duration_ns=10, stats=None):
        self.name = name
        self.start_ns = start_ns
        self.duration_ns = duration_ns
        self.stats = stats or []

    class _FakeLine:

      def __init__(self, name, events=None):
        self.name = name
        self.events = events or []

    class _FakePlane:

      def __init__(self, name, lines=None):
        self.name = name
        self.lines = lines or []

    fake_event1 = _FakeEvent("event1", 0, 10)
    fake_event2 = _FakeEvent("event2", 10, 10)
    fake_line = _FakeLine("line1", [fake_event1, fake_event2])
    fake_plane = _FakePlane("plane1", [fake_line])

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ):
      # max_events=-1 should return all 2 events, not 1
      res_unlimited = xplane_tools.list_xplane_events(
          "test_session", max_events=-1
      )
      self.assertIn("event1", res_unlimited)
      self.assertIn("event2", res_unlimited)

      # max_events=1 should return only 1 event
      res_one = xplane_tools.list_xplane_events("test_session", max_events=1)
      self.assertIn("event1", res_one)
      self.assertNotIn("event2", res_one)


if __name__ == "__main__":
  absltest.main()
