"""Unit tests for OSS hlo_tools in XProf CLI."""

import json
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

  def test_list_hlo_modules_empty(self):
    """Verifies list_hlo_modules returns JSON with count 0 when no files exist."""
    with mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[]):
      result = hlo_tools.list_hlo_modules("empty_session")
      parsed = json.loads(result)
      self.assertEqual(parsed["status"], "SUCCESS")
      self.assertEqual(parsed["count"], 0)
      self.assertEqual(parsed["modules"], [])

  def test_list_hlo_modules_success(self):
    """Verifies list_hlo_modules lists module names cleanly in JSON."""
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    f2 = self.session_dir / "module_0002.jit_eval.hlo_proto.pb"
    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[f1, f2]
    ):
      result = hlo_tools.list_hlo_modules(str(self.session_dir))
      parsed = json.loads(result)
      self.assertEqual(parsed["status"], "SUCCESS")
      self.assertEqual(parsed["count"], 2)
      self.assertEqual(
          parsed["modules"],
          ["module_0001.jit_compute", "module_0002.jit_eval"],
      )

  def test_get_hlo_module_content_success(self):
    """Verifies get_hlo_module_content retrieves and wraps text in JSON."""
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
      parsed = json.loads(content)
      self.assertEqual(parsed["status"], "SUCCESS")
      self.assertEqual(parsed["module_name"], "module_0001.jit_compute")
      self.assertIn("HloModule jit_compute", parsed["content"])
      self.assertIn("%neg = f32[10] negate(%x)", parsed["content"])

  def test_get_hlo_module_content_missing_module_raises_value_error(self):
    """Verifies missing module raises ValueError."""
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[f1]
    ):
      with self.assertRaisesRegex(ValueError, "Module 'nonexistent' not found"):
        hlo_tools.get_hlo_module_content(
            str(self.session_dir), module_name="nonexistent"
        )

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

  def test_get_hlo_neighborhood_op_name_alias(self):
    """Verifies get_hlo_neighborhood supports op_name alias."""
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
          str(self.session_dir), op_name="mul", radius=1
      )
      self.assertIn("%mul", neighborhood)
      self.assertIn("%x", neighborhood)

  def test_get_hlo_neighborhood_missing_name_raises_value_error(self):
    """Verifies ValueError when neither instruction_name nor op_name given."""
    with self.assertRaises(ValueError):
      hlo_tools.get_hlo_neighborhood(str(self.session_dir))

  def test_get_hlo_neighborhood_conflicting_names_raises_value_error(self):
    """Verifies ValueError when instruction_name and op_name conflict."""
    with self.assertRaisesRegex(ValueError, "Conflicting arguments"):
      hlo_tools.get_hlo_neighborhood(
          str(self.session_dir), instruction_name="mul", op_name="add"
      )

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
      parsed = json.loads(content)
      self.assertEqual(parsed["status"], "SUCCESS")
      self.assertEqual(parsed["content"], "HloModule test_export\n")
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

  def test_resolve_module_name_base_and_prefix(self):
    f1 = self.session_dir / "jit_train_step(7216021599878099202).hlo_proto.pb"
    f2 = self.session_dir / "eval_step(1111111111).hlo_proto.pb"
    f1.write_bytes(b"dummy")
    f2.write_bytes(b"dummy")

    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[f1, f2]
    ):
      self.assertEqual(
          hlo_tools.resolve_module_name(
              str(self.session_dir), "jit_train_step"
          ),
          "jit_train_step(7216021599878099202)",
      )
      self.assertEqual(
          hlo_tools.resolve_module_name(str(self.session_dir), "eval"),
          "eval_step(1111111111)",
      )
      with self.assertRaises(ValueError):
        hlo_tools.resolve_module_name(str(self.session_dir), "missing_mod")


if __name__ == "__main__":
  absltest.main()
