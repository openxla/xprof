"""Tests for get_llo_report_tool module."""

import json
from unittest import mock
from absl.testing import absltest
from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import
from xprof.cli.tools import get_llo_report_tool


class GetLloReportToolTest(absltest.TestCase):

  def _create_mock_xspace(self) -> bytes:
    xspace = xplane_pb2.XSpace()
    plane = xspace.planes.add()
    plane.name = "/device:TPU:0"
    plane.stat_metadata[1].id = 1
    plane.stat_metadata[1].name = "details"
    line = plane.lines.add()
    line.name = "MXU Instructions"
    event = line.events.add()
    event.metadata_id = 10
    event.offset_ps = 1000
    event.duration_ps = 5000
    plane.event_metadata[10].id = 10
    plane.event_metadata[10].name = "vmatmul"
    return xspace.SerializeToString()

  @mock.patch(
      "google3.third_party.xprof.plugin.xprof.cli.internal.google.xprof_client.get_client"
  )
  def test_get_llo_report_json(self, mock_get_client):
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_hosts.return_value = ["host1"]
    mock_client.get_serialized_xspace.return_value = self._create_mock_xspace()

    res_str = get_llo_report_tool.get_llo_report(
        session_id="test_session", host="host1", format="json"
    )
    res = json.loads(res_str)

    self.assertEqual(res.get("title"), "XprofLloRuntimeReport")
    self.assertEqual(res["report_metadata"]["session_id"], "test_session")
    self.assertEqual(res["report_metadata"]["total_event_count"], 1)

  @mock.patch(
      "google3.third_party.xprof.plugin.xprof.cli.internal.google.xprof_client.get_client"
  )
  def test_get_llo_report_markdown(self, mock_get_client):
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_hosts.return_value = ["host1"]
    mock_client.get_serialized_xspace.return_value = self._create_mock_xspace()

    res_str = get_llo_report_tool.get_llo_report(
        session_id="test_session", host="host1", format="markdown"
    )
    self.assertIn("# LLO Runtime Execution & Bottleneck Report", res_str)
    self.assertIn("## Executive Summary", res_str)

  @mock.patch(
      "google3.third_party.xprof.plugin.xprof.cli.internal.google.xprof_client.get_client"
  )
  def test_get_llo_report_invalid_host(self, mock_get_client):
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_hosts.return_value = ["host1"]

    res_str = get_llo_report_tool.get_llo_report(
        session_id="test_session", host="invalid_host", format="json"
    )
    res = json.loads(res_str)
    self.assertIn("error", res)
    self.assertIn("Invalid host", res["error"])


if __name__ == "__main__":
  absltest.main()
