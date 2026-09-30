"""Unit tests for llo_static_analysis_tool CLI interface in OSS."""

import json
from unittest import mock

from absl.testing import absltest
from xprof.cli.tools.oss import llo_static_analysis_tool


class OssLloStaticAnalysisToolTest(absltest.TestCase):

  def test_host_filtering_is_rejected(self):
    payload = json.loads(
        llo_static_analysis_tool.get_llo_static_analysis(
            "/tmp/trace.xplane.pb", mode="opcodes", host="host0"
        )
    )

    self.assertEqual(payload["status"], "ERROR")
    self.assertEqual(payload["mode"], "opcodes")
    self.assertIn("not supported in OSS builds", payload["error"])

  def test_directory_without_xplane_files_is_an_error(self):
    empty_dir = self.create_tempdir().full_path

    payload = json.loads(
        llo_static_analysis_tool.get_llo_static_analysis(
            empty_dir, mode="region_tree"
        )
    )

    self.assertEqual(payload["status"], "ERROR")
    self.assertIn("No *.xplane.pb files found", payload["error"])

  def test_directory_resolves_to_the_first_xplane_file(self):
    trace_dir = self.create_tempdir()
    trace = trace_dir.create_file("host0.xplane.pb", content="")

    with mock.patch.object(
        llo_static_analysis_tool,
        "_pywrap_profiler_plugin",
        autospec=True,
        spec_set=True,
    ) as mock_pywrap:
      mock_pywrap.get_llo_static_analysis_json.return_value = "{}"
      llo_static_analysis_tool.get_llo_static_analysis(
          trace_dir.full_path, mode="bundle_util", hlo_op="fusion.1", bundle=3
      )

    mock_pywrap.get_llo_static_analysis_json.assert_called_once_with(
        trace.full_path, mode="bundle_util", hlo_op="fusion.1", bundle=3
    )

  def test_builds_without_embedded_features_report_the_error(self):
    with mock.patch.object(
        llo_static_analysis_tool,
        "_pywrap_profiler_plugin",
        autospec=True,
        spec_set=True,
    ) as mock_pywrap:
      mock_pywrap.get_llo_static_analysis_json.side_effect = (
          NotImplementedError("Built without embedded LLO analysis support.")
      )
      payload = json.loads(
          llo_static_analysis_tool.get_llo_static_analysis(
              "/tmp/trace.xplane.pb", mode="spills"
          )
      )

    self.assertEqual(payload["status"], "ERROR")
    self.assertEqual(payload["semantics"], "static_modelled_schedule")
    self.assertEqual(
        payload["error"], "Built without embedded LLO analysis support."
    )


if __name__ == "__main__":
  absltest.main()
