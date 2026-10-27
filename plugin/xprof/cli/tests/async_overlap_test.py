"""Unit tests for async_overlap and analyze_async_overlap_tool."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import async_overlap
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xplane_tools
from xprof.cli.tools import analyze_async_overlap_tool


class AsyncOverlapTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(decorators.Cache, instance=True)
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(decorators, "get_cache", return_value=mock_cache)
    )

  def test_instruction_name_and_roles(self):
    self.assertEqual(
        async_overlap.instruction_name(
            "%all-gather-start.3 = (f32[8]) all-gather-start(f32[4] %p0)"
        ),
        "all-gather-start.3",
    )
    self.assertEqual(
        async_overlap.async_role("all-gather-start.3"),
        ("all-gather.3", "start"),
    )
    self.assertEqual(
        async_overlap.async_role("all-gather-done.3"),
        ("all-gather.3", "done"),
    )
    self.assertEqual(
        async_overlap.async_role("send.2"),
        ("send.2", "start"),
    )
    self.assertEqual(
        async_overlap.async_role("send-done.2"),
        ("send.2", "done"),
    )
    self.assertIsNone(async_overlap.async_role("fusion.12"))

  @parameterized.parameters(
      ("all-gather-start.3", "all-gather"),
      ("all-gather.3-start", "all-gather"),
      ("all-gather-done.3", "all-gather"),
      ("copy-start.1", "copy"),
      ("copy-done", "copy"),
      ("send.2", "send"),
      ("recv-done.4", "recv"),
      ("fusion.7.call-start", "call"),
      ("fusion.7.call-done", "call"),
      ("fusion.7.call.2-start", "call"),
      ("all-reduce", "all-reduce"),
  )
  def test_op_family_takes_last_non_numeric_token(self, name, family):
    self.assertEqual(async_overlap.op_family(name), family)

  def test_analyze_keeps_hosts_with_the_same_plane_apart(self):
    # Both hosts have `/device:TPU:0`. Host a's compute [0, 100) must not
    # cover host b's window [0, 100), and a start on one host must not pair
    # with a done on the other.
    tc = "/device:TPU:0"
    sc = "/device:TPU:0 SparseCore 0"

    def event(host, plane, line, name, start_ps, duration_ps):
      return async_overlap.TimelineEvent(
          plane=plane,
          line=line,
          name=name,
          start_ps=start_ps,
          duration_ps=duration_ps,
          host=host,
      )

    events = [
        event("host-a", tc, "XLA Ops", "fusion.1", 0, 100),
        event("host-a", tc, "XLA Ops", "all-gather-start.1", 200, 10),
        event("host-a", sc, "Sparse Core Ops", "sc_kernel.1", 0, 100),
        event("host-b", tc, "XLA Ops", "copy-start.1", 0, 10),
        event("host-b", tc, "XLA Ops", "copy-done.1", 90, 10),
        event("host-b", tc, "XLA Ops", "all-gather-done.1", 300, 10),
        event("host-b", sc, "Sparse Core Ops", "sc_kernel.2", 150, 50),
    ]
    res = async_overlap.analyze(events)

    self.assertEqual(res["summary"]["count"], 1)
    self.assertEqual(res["summary"]["unpaired_ops"], 2)
    self.assertEqual(res["summary"]["exposed"], 1)
    self.assertEqual(list(res["by_plane"]), ["host-b:/device:TPU:0"])
    self.assertEqual(res["by_plane"]["host-b:/device:TPU:0"]["host"], "host-b")
    self.assertEqual(res["top_exposed"][0]["host"], "host-b")
    self.assertEqual(res["top_exposed"][0]["plane"], tc)
    self.assertEqual(res["top_exposed"][0]["exposed_ps"], 100)
    self.assertEqual(
        [
            (c["host"], c["pairs"][0]["concurrent_ps"])
            for c in res["concurrency"]
        ],
        [("host-a", 100), ("host-b", 0)],
    )

  def test_analyze_single_host_output_has_no_host_keys(self):
    res = async_overlap.analyze([
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="Async XLA Ops",
            name="copy-start.1",
            start_ps=0,
            duration_ps=100,
        ),
    ])
    self.assertEqual(list(res["by_plane"]), ["/device:TPU:0"])
    self.assertNotIn("host", res["by_plane"]["/device:TPU:0"])
    self.assertNotIn("host", res["top_exposed"][0])

  def test_analyze_with_async_ops_line_classifies_hidden_partial_exposed(self):
    events = [
        # Compute ops on XLA Ops: [100, 200) and [300, 350).
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="%fusion.1 = f32[8] fusion()",
            start_ps=100,
            duration_ps=100,
        ),
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="%fusion.2 = f32[8] fusion()",
            start_ps=300,
            duration_ps=50,
        ),
        # Async windows on Async XLA Ops:
        # 1) hidden: [100, 200) -> 100% covered
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="Async XLA Ops",
            name="%all-gather-start.1 = all-gather-start()",
            start_ps=100,
            duration_ps=100,
        ),
        # 2) partial: [300, 400) -> 50% covered
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="Async XLA Ops",
            name="%all-reduce-start.1 = all-reduce-start()",
            start_ps=300,
            duration_ps=100,
        ),
        # 3) exposed: [500, 600) -> 0% covered
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="Async XLA Ops",
            name="%copy-start.1 = copy-start()",
            start_ps=500,
            duration_ps=100,
        ),
    ]
    res = async_overlap.analyze(events)
    self.assertEqual(res["summary"]["count"], 3)
    self.assertEqual(res["summary"]["span_ps"], 300)
    self.assertEqual(res["summary"]["exposed_ps"], 150)
    self.assertEqual(res["summary"]["hidden"], 1)
    self.assertEqual(res["summary"]["partial"], 1)
    self.assertEqual(res["summary"]["exposed"], 1)
    self.assertAlmostEqual(res["summary"]["overlap_ratio"], 0.5)
    self.assertEqual(
        res["by_plane"]["/device:TPU:0"]["window_source"], "Async XLA Ops"
    )
    self.assertEqual(
        [r["name"] for r in res["top_exposed"]],
        ["copy-start.1", "all-reduce-start.1", "all-gather-start.1"],
    )
    self.assertEqual(
        list(res["by_op"].keys()),
        ["copy", "all-reduce", "all-gather"],
    )

  def test_analyze_fallback_pairs_start_and_done_excluding_bookends(self):
    events = [
        # Start op [0, 10), compute [10, 90), done op [90, 100), plus an
        # unpaired start op.
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="all-to-all-start.1",
            start_ps=0,
            duration_ps=10,
        ),
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="fusion.1",
            start_ps=10,
            duration_ps=80,
        ),
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="all-to-all-done.1",
            start_ps=90,
            duration_ps=10,
        ),
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="copy-start.9",
            start_ps=200,
            duration_ps=10,
        ),
    ]
    res = async_overlap.analyze(events)
    self.assertEqual(res["summary"]["count"], 1)
    self.assertEqual(res["summary"]["unpaired_ops"], 1)
    self.assertEqual(res["summary"]["span_ps"], 100)
    # Only fusion.1 ([10, 90)) counts as compute coverage; start/done bookends
    # are excluded -> 80 / 100 = 0.8 (partial).
    self.assertEqual(res["summary"]["exposed_ps"], 20)
    self.assertEqual(res["summary"]["partial"], 1)
    self.assertEqual(
        res["by_plane"]["/device:TPU:0"]["window_source"], "start_done_pairs"
    )

  def test_cross_engine_concurrency_per_tpu_chip(self):
    events = [
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="XLA Ops",
            name="fusion.1",
            start_ps=0,
            duration_ps=100,
        ),
        async_overlap.TimelineEvent(
            plane="/device:TPU:0 SparseCore 0",
            line="Sparse Core Ops",
            name="sc_kernel.1",
            start_ps=50,
            duration_ps=50,
        ),
    ]
    res = async_overlap.analyze(events)
    self.assertLen(res["concurrency"], 1)
    chip0 = res["concurrency"][0]
    self.assertEqual(chip0["chip"], 0)
    self.assertLen(chip0["pairs"], 1)
    self.assertEqual(chip0["pairs"][0]["concurrent_ps"], 50)
    self.assertAlmostEqual(chip0["pairs"][0]["concurrent_ratio"], 1.0)

  @parameterized.parameters(
      (-0.1, 0.9),
      (0.5, 0.5),
      (0.8, 0.2),
      (0.1, 1.1),
  )
  def test_invalid_thresholds_raise_value_error(self, exposed, hidden):
    with self.assertRaises(ValueError):
      async_overlap.analyze(
          [], hidden_threshold=hidden, exposed_threshold=exposed
      )
    with self.assertRaises(ValueError):
      analyze_async_overlap_tool.analyze_async_overlap(
          "session_1", hidden_threshold=hidden, exposed_threshold=exposed
      )

  def test_tool_returns_no_device_timeline_events_when_empty(self):
    with mock.patch.object(
        xplane_tools, "iter_timeline_events", return_value=[]
    ):
      out = json.loads(
          analyze_async_overlap_tool.analyze_async_overlap("empty_session")
      )
    self.assertEqual(out["status"], "NO_DEVICE_TIMELINE_EVENTS")
    self.assertEqual(out["session_id"], "empty_session")
    self.assertEqual(out["summary"]["count"], 0)

  def test_tool_returns_ok_envelope(self):
    sample = [
        async_overlap.TimelineEvent(
            plane="/device:TPU:0",
            line="Async XLA Ops",
            name="copy-start.1",
            start_ps=0,
            duration_ps=100,
        ),
    ]
    with mock.patch.object(
        xplane_tools, "iter_timeline_events", return_value=sample
    ):
      out = json.loads(
          analyze_async_overlap_tool.analyze_async_overlap(
              "sess_ok", host="host-0"
          )
      )
    self.assertEqual(out["status"], "OK")
    self.assertEqual(out["session_id"], "sess_ok")
    self.assertEqual(out["host"], "host-0")
    self.assertEqual(out["summary"]["count"], 1)
    self.assertEqual(out["summary"]["exposed"], 1)


if __name__ == "__main__":
  absltest.main()
