"""Unit tests for OSS xplane_tools.

Covers `iter_planes` (including the fixes for the two review comments on
cl/954632105: non-existent absolute paths raise FileNotFoundError instead of
silently bypassing the XProf server fallback, and directory globs search for
both .xplane.pb and .xspace.pb files) and `inspect_capture`, whose 1P
counterpart is tested in tests/google/xplane_tools_test.py.
"""

import json
import pathlib
import tempfile
from unittest import mock

import fire

from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xplane_tools


class _FakeEvent:

  def __init__(self, name, start_ns=0, duration_ns=10, stats=None):
    self.name = name
    self.start_ns = start_ns
    self.duration_ns = duration_ns
    self.stats = stats or []


class _FakeLine:

  def __init__(self, name, events=None):
    self.name = name
    self.display_name = ""
    self.events = events or []


class _FakeStatMetadata:

  def __init__(self, name):
    self.name = name


class _FakePlane:

  def __init__(self, name, lines=None):
    self.name = name
    self.lines = lines or []
    self.stats = []
    self.stat_metadata = {}


class _FakeSpace:

  def __init__(self, planes):
    self.planes = planes


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

  def test_logdir_root_resolves_to_latest_run_only(self):
    """iter_planes on a logdir root should yield planes only from the latest run."""
    with tempfile.TemporaryDirectory() as tmpdir:
      logdir = pathlib.Path(tmpdir) / "logs"
      old_run = logdir / "plugins" / "profile" / "2026_09_16_10_00_00"
      latest_run = logdir / "plugins" / "profile" / "2026_09_16_12_00_00"
      old_run.mkdir(parents=True)
      latest_run.mkdir(parents=True)
      (old_run / "host.xplane.pb").write_bytes(b"old_bytes")
      (latest_run / "host.xplane.pb").write_bytes(b"latest_bytes")

      def fake_from_serialized(raw_bytes):
        pd = mock.MagicMock()
        pd.planes = (
            ["old_plane"] if raw_bytes == b"old_bytes" else ["latest_plane"]
        )
        return pd

      with mock.patch.object(
          xplane_tools.profiler.ProfileData,
          "from_serialized_xspace",
          side_effect=fake_from_serialized,
      ):
        planes = list(xplane_tools.iter_planes(str(logdir)))

      self.assertEqual(planes, ["latest_plane"])


class InspectCaptureTest(absltest.TestCase):
  """Mirrors tests/google/xplane_tools_test.py so 1P and 3P stay in sync."""

  def test_inspect_capture_gates(self):
    fake_plane = _FakePlane(
        "/device:TPU:0",
        lines=[
            _FakeLine("XLA Ops", [_FakeEvent("e0")]),
            _FakeLine("Pallas Primitives", [_FakeEvent("e1")]),
            _FakeLine("TC Stats", [_FakeEvent("e2")]),
        ],
    )

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ):
      res = json.loads(xplane_tools.inspect_capture("session_inspect"))

    self.assertTrue(res["capture_valid_for_device_analysis"])
    self.assertTrue(res["capture_valid_for_llo_analysis"])
    self.assertTrue(res["capture_valid_for_counter_sampling"])
    self.assertLen(res["planes"], 1)
    self.assertEqual(res["planes"][0]["total_events"], 3)

  def test_inspect_capture_host_metadata_llo_proto(self):
    fake_plane = _FakePlane("/host:metadata")
    fake_plane.stat_metadata = {1: _FakeStatMetadata("llo_proto")}

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ):
      res = json.loads(xplane_tools.inspect_capture("session_llo_meta"))

    self.assertFalse(res["capture_valid_for_device_analysis"])
    self.assertTrue(res["capture_valid_for_llo_analysis"])
    self.assertFalse(res["capture_valid_for_counter_sampling"])

  def test_inspect_capture_host_metadata_llo_proto_plane_stat(self):
    """`llo_proto` carried as a plane-level stat also gates LLO analysis."""
    fake_plane = _FakePlane("/host:metadata")
    fake_plane.stats = [("llo_proto", b"")]

    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ):
      res = json.loads(xplane_tools.inspect_capture("session_llo_stat"))

    self.assertTrue(res["capture_valid_for_llo_analysis"])

  def test_list_xplane_events_window_semantics_and_filter(self):
    fake_plane = _FakePlane(
        "/device:TPU:0",
        lines=[
            _FakeLine(
                "XLA Ops",
                events=[
                    _FakeEvent("Fusion.1", start_ns=80, duration_ns=70),
                    _FakeEvent("Copy.1", start_ns=110, duration_ns=100),
                ],
            )
        ],
    )
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ):
      contained = json.loads(
          xplane_tools.list_xplane_events(
              "sess_oss_win_c",
              start_time_ps=100_000,
              end_time_ps=250_000,
              window_semantics="containment",
          )
      )
      overlapped = json.loads(
          xplane_tools.list_xplane_events(
              "sess_oss_win_o",
              start_time_ps=100_000,
              end_time_ps=250_000,
              window_semantics="overlap",
          )
      )
      filtered = json.loads(
          xplane_tools.list_xplane_events(
              "sess_oss_flt",
              min_duration_ps=70_000,
              filter='event =~ "^Fusion" AND duration_ns == 70',
          )
      )
    self.assertEqual([e["event"] for e in contained["events"]], ["Copy.1"])
    self.assertEqual(
        [e["event"] for e in overlapped["events"]], ["Fusion.1", "Copy.1"]
    )
    self.assertEqual([e["event"] for e in filtered["events"]], ["Fusion.1"])

  def test_aggregate_xplane_events_percentiles_and_group_by(self):
    fake_plane = _FakePlane(
        "/device:TPU:0",
        lines=[
            _FakeLine(
                "XLA Ops",
                events=[
                    _FakeEvent("Fusion", start_ns=i * 10, duration_ns=d)
                    for i, d in enumerate((1, 2, 3, 4, 5))
                ],
            )
        ],
    )
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ):
      payload = json.loads(
          xplane_tools.aggregate_xplane_events(
              "sess_oss_agg",
              group_by="plane,line,event",
              filter="duration_ns >= 1",
          )
      )
    self.assertEqual(payload["returned"], 1)
    row = payload["aggregates"][0]
    self.assertEqual(row["plane"], "/device:TPU:0")
    self.assertEqual(row["line"], "XLA Ops")
    self.assertEqual(row["event"], "Fusion")
    self.assertEqual(row["count"], 5)
    self.assertEqual(row["min_duration_ps"], 1000)
    self.assertAlmostEqual(row["p50_duration_ps"], 3000.0)
    self.assertAlmostEqual(row["p90_duration_ps"], 4600.0)
    self.assertAlmostEqual(row["p99_duration_ps"], 4960.0)
    self.assertEqual(row["max_duration_ps"], 5000)

  def test_aggregate_xplane_events_fire_comma_separated_group_by(self):
    # python-fire parses `--group_by=plane,event` into a tuple.
    fake_plane = _FakePlane(
        "/device:TPU:0",
        lines=[
            _FakeLine(
                "XLA Ops",
                events=[
                    _FakeEvent("Fusion", start_ns=0, duration_ns=1),
                    _FakeEvent("Copy", start_ns=10, duration_ns=2),
                ],
            )
        ],
    )
    with mock.patch.object(
        xplane_tools, "iter_planes", return_value=[fake_plane]
    ), mock.patch("sys.stdout"):
      payload = json.loads(
          fire.Fire(
              xplane_tools.aggregate_xplane_events,
              command=[
                  "sess_oss_agg_fire",
                  "--group_by=plane,event",
                  "--bypass_cache=True",
              ],
          )
      )
    self.assertEqual(payload["returned"], 2)
    self.assertEqual(
        sorted((r["plane"], r["event"]) for r in payload["aggregates"]),
        [("/device:TPU:0", "Copy"), ("/device:TPU:0", "Fusion")],
    )
    self.assertNotIn("line", payload["aggregates"][0])

  def test_iter_timeline_events_filters_and_converts_ns_to_ps(self):
    fake_plane = _FakePlane(
        "/device:TPU:0",
        lines=[
            _FakeLine(
                "Async XLA Ops",
                events=[_FakeEvent("copy-start.1", start_ns=5, duration_ns=2)],
            ),
            _FakeLine("XLA Modules", events=[_FakeEvent("mod", start_ns=1)]),
        ],
    )
    events = xplane_tools.iter_timeline_events(_FakeSpace([fake_plane]))
    self.assertLen(events, 1)
    self.assertEqual(events[0].plane, "/device:TPU:0")
    self.assertEqual(events[0].line, "Async XLA Ops")
    self.assertEqual(events[0].name, "copy-start.1")
    self.assertEqual(events[0].start_ps, 5000)
    self.assertEqual(events[0].duration_ps, 2000)
    self.assertEqual(events[0].host, "")

  def _write_two_host_traces(self, tmpdir: str) -> None:
    (pathlib.Path(tmpdir) / "host-0.xplane.pb").write_bytes(b"host-0")
    (pathlib.Path(tmpdir) / "host-1.xplane.pb").write_bytes(b"host-1")

  def _fake_from_serialized(self, raw_bytes):
    """Returns one `/device:TPU:0` plane whose event names its trace."""
    name = "copy-start.1" if raw_bytes == b"host-0" else "copy-done.1"
    pd = mock.MagicMock()
    pd.planes = [
        _FakePlane(
            "/device:TPU:0", lines=[_FakeLine("XLA Ops", [_FakeEvent(name)])]
        )
    ]
    return pd

  def test_iter_timeline_events_labels_each_trace_of_a_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_two_host_traces(tmpdir)
      with mock.patch.object(
          xplane_tools.profiler.ProfileData,
          "from_serialized_xspace",
          side_effect=self._fake_from_serialized,
      ):
        events = xplane_tools.iter_timeline_events(tmpdir)
    self.assertCountEqual(
        [(e.host, e.plane, e.name) for e in events],
        [
            ("host-0", "/device:TPU:0", "copy-start.1"),
            ("host-1", "/device:TPU:0", "copy-done.1"),
        ],
    )

  def test_iter_timeline_events_host_selects_one_trace_of_a_directory(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      self._write_two_host_traces(tmpdir)
      with mock.patch.object(
          xplane_tools.profiler.ProfileData,
          "from_serialized_xspace",
          side_effect=self._fake_from_serialized,
      ):
        events = xplane_tools.iter_timeline_events(tmpdir, host="host-1")
        with self.assertRaisesRegex(ValueError, "Unknown host"):
          xplane_tools.iter_timeline_events(tmpdir, host="host-2")
    self.assertEqual([(e.host, e.name) for e in events], [("", "copy-done.1")])

  def test_iter_timeline_events_rejects_host_for_in_memory_source(self):
    with self.assertRaisesRegex(ValueError, "host is only supported"):
      xplane_tools.iter_timeline_events(_FakeSpace([]), host="host-0")

  def _fake_named_from_serialized(self, raw_bytes):
    """Returns one `/device:TPU:0` plane with one event named `raw_bytes`."""
    pd = mock.MagicMock()
    pd.planes = [
        _FakePlane(
            "/device:TPU:0",
            lines=[_FakeLine("XLA Ops", [_FakeEvent(raw_bytes.decode())])],
        )
    ]
    return pd

  def _timeline(self, session_id, **kwargs):
    with mock.patch.object(
        xplane_tools.profiler.ProfileData,
        "from_serialized_xspace",
        side_effect=self._fake_named_from_serialized,
    ):
      events = xplane_tools.iter_timeline_events(session_id, **kwargs)
    return [(e.host, e.name) for e in events]

  def test_iter_timeline_events_file_with_host_shared_by_a_sibling(self):
    # The CLI resolves `--host=a.h0` to the file `a.h0.xplane.pb` and passes
    # its `get_hosts` name "h0", which `b.h0.xplane.pb` also has.
    with tempfile.TemporaryDirectory() as tmpdir:
      a_path = pathlib.Path(tmpdir) / "a.h0.xplane.pb"
      a_path.write_bytes(b"a")
      (pathlib.Path(tmpdir) / "b.h0.xplane.pb").write_bytes(b"b")
      self.assertEqual(self._timeline(str(a_path), host="h0"), [("", "a")])
      self.assertEqual(self._timeline(str(a_path), host="a.h0"), [("", "a")])
      with self.assertRaisesRegex(ValueError, "Conflicting selection"):
        self._timeline(str(a_path), host="b.h0")

  def test_iter_timeline_events_reads_file_without_trace_suffix(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = pathlib.Path(tmpdir) / "trace.pb"
      path.write_bytes(b"trace")
      (pathlib.Path(tmpdir) / "host-0.xplane.pb").write_bytes(b"host-0")
      self.assertEqual(self._timeline(str(path)), [("", "trace")])

  def test_iter_timeline_events_labels_duplicate_stems_by_relative_path(self):
    for names in (
        ("x/host-0.xplane.pb", "y/host-0.xplane.pb"),
        ("host-0.xplane.pb", "host-0.xspace.pb"),
    ):
      with self.subTest(names=names):
        with tempfile.TemporaryDirectory() as tmpdir:
          for name in names:
            path = pathlib.Path(tmpdir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
          self.assertCountEqual(
              self._timeline(tmpdir), [(name, name) for name in names]
          )


if __name__ == "__main__":
  absltest.main()
