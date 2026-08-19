"""Unit tests for OSS hlo_tools in XProf CLI."""

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from xprof.cli.internal.oss import hlo_tools
from xprof.cli.internal.oss import xprof_client


class OssHloToolsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()
    self.session_dir = pathlib.Path(self.test_dir.name)

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_debug_info_collection_container(self):
    """Verifies fallback _DebugInfoCollection structure."""
    container = hlo_tools._DebugInfoCollection(
        hlo_proto=["proto1"], program_id=["prog1"]
    )
    self.assertEqual(container.hlo_proto, ["proto1"])
    self.assertEqual(container.program_id, ["prog1"])

  def test_list_hlo_modules_empty(self):
    """Verifies list_hlo_modules returns friendly message when no files exist."""
    with mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[]):
      result = hlo_tools.list_hlo_modules("empty_session")
      self.assertIn("No HLO modules found", result)

  def test_list_hlo_modules_success(self):
    """Verifies list_hlo_modules lists module names cleanly."""
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    f2 = self.session_dir / "module_0002.jit_eval.hlo_proto.pb"
    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[f1, f2]
    ):
      result = hlo_tools.list_hlo_modules(str(self.session_dir))
      self.assertIn("Found 2 HLO modules:", result)
      self.assertIn("0. module_0001.jit_compute", result)
      self.assertIn("1. module_0002.jit_eval", result)

  def test_get_hlo_module_content_success(self):
    """Verifies get_hlo_module_content retrieves and truncates text."""
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    mock_client = mock.MagicMock()
    sample_hlo = (
        "HloModule jit_compute\n\n%entry (\n  %x = f32[10] parameter(0)\n  ROOT"
        " %neg = f32[10] negate(%x)\n)\n"
    )
    mock_client.fetch.return_value = (None, sample_hlo.encode("utf-8"))

    with (
        mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[f1]),
        mock.patch.object(xprof_client, "get_client", return_value=mock_client),
    ):
      content = hlo_tools.get_hlo_module_content(
          str(self.session_dir), module_name="module_0001.jit_compute"
      )
      self.assertIn("HloModule jit_compute", content)
      self.assertIn("%neg = f32[10] negate(%x)", content)

  def test_get_hlo_neighborhood_bfs(self):
    """Verifies get_hlo_neighborhood traverses operands and users."""
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    mock_client = mock.MagicMock()
    hlo_graph = (
        "%entry (\n"
        "  %x = f32[10] parameter(0)\n"
        "  %w = f32[10] parameter(1)\n"
        "  %mul = f32[10] multiply(%x, %w)\n"
        "  %add = f32[10] add(%mul, %x)\n"
        "  ROOT %neg = f32[10] negate(%add)\n"
        ")\n"
    )
    mock_client.fetch.return_value = (None, hlo_graph.encode("utf-8"))

    with (
        mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[f1]),
        mock.patch.object(xprof_client, "get_client", return_value=mock_client),
    ):
      neighborhood = hlo_tools.get_hlo_neighborhood(
          str(self.session_dir), instruction_name="mul", radius=1
      )
      self.assertIn("%mul", neighborhood)
      self.assertIn("%x", neighborhood)
      self.assertIn("%w", neighborhood)

  def test_get_hlo_text_file_export(self):
    """Verifies get_hlo_text saves output to file path when requested."""
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    mock_client = mock.MagicMock()
    mock_client.fetch.return_value = (None, b"HloModule test_export\n")
    out_file = self.session_dir / "exported_hlo.txt"

    with (
        mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[f1]),
        mock.patch.object(xprof_client, "get_client", return_value=mock_client),
    ):
      content = hlo_tools.get_hlo_text(
          str(self.session_dir), path=str(out_file)
      )
      self.assertEqual(content, "HloModule test_export\n")
      self.assertTrue(out_file.exists())
      self.assertEqual(
          out_file.read_text(encoding="utf-8"), "HloModule test_export\n"
      )

  def test_get_hlo_proto_files_finds_nested_hlo_protos(self):
    """Verifies _get_hlo_proto_files discovers HLO protos in nested subdirs."""
    nested_dir = (
        self.session_dir / "plugins" / "profile" / "2026_08_18_01_02_03"
    )
    nested_dir.mkdir(parents=True, exist_ok=True)
    f1 = nested_dir / "module_nested.hlo_proto.pb"
    f1.write_bytes(b"dummy")

    mock_client = mock.MagicMock()
    mock_client.get_run_dir.return_value = self.session_dir

    with (
        mock.patch.object(xprof_client, "get_client", return_value=mock_client),
        mock.patch.object(hlo_tools, "generate_hlo_protos"),
    ):
      files = hlo_tools._get_hlo_proto_files(str(self.session_dir))
      self.assertEqual(files, [f1])


if __name__ == "__main__":
  absltest.main()
