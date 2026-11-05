"""Unit tests for OSS hlo_tools in XProf CLI."""

import json
import pathlib
import tempfile
from unittest import mock

from google.protobuf import descriptor_pool
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

  def test_get_hlo_text_truncation_metadata(self):
    f1 = self.session_dir / "module_1.hlo_proto.pb"
    f1.write_bytes(b"dummy")
    full_hlo = "\n".join(f"line_{i}" for i in range(100))
    mock_client = mock.MagicMock()
    mock_client.get_run_dir.return_value = self.session_dir
    mock_client.fetch.return_value = (None, full_hlo.encode("utf-8"))

    with (
        mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[f1]),
        mock.patch.object(xprof_client, "get_client", return_value=mock_client),
    ):
      content = hlo_tools.get_hlo_text(str(self.session_dir), max_lines=25)
      parsed = json.loads(content)
      self.assertEqual(parsed["status"], "SUCCESS")
      self.assertTrue(parsed["truncated"])
      self.assertEqual(parsed["line_count"], 100)
      self.assertEqual(parsed["returned_line_count"], 25)
      expected_bytes = len(full_hlo.encode("utf-8"))
      self.assertEqual(parsed["byte_count"], expected_bytes)
      self.assertIn("[Truncated to 25 of 100 lines", parsed["content"])

  def test_oss_query_hlo_graph_and_directional_neighborhood(self):
    f1 = self.session_dir / "module_0001.jit_compute.hlo_proto.pb"
    mock_client = mock.MagicMock()
    hlo_graph = (
        "%fused_comp (\n"
        "  %p0 = bf16[4,128] parameter(0)\n"
        "  ROOT %sub_add = bf16[4,128] add(bf16[4,128] %p0, bf16[4,128] %p0)\n"
        ")\n\n"
        "ENTRY %main (\n"
        "  %x = bf16[4,128] parameter(0)\n"
        "  %bc = bf16[4,128] bitcast(bf16[4,128] %x)\n"
        "  %fus = bf16[4,128] fusion(bf16[4,128] %bc),"
        " kind=kLoop, calls=%fused_comp\n"
        "  ROOT %out = bf16[4,128] custom-call(bf16[4,128] %fus)\n"
        ")\n"
    )
    mock_client.fetch.return_value = (None, hlo_graph.encode("utf-8"))

    with (
        mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[f1]),
        mock.patch.object(xprof_client, "get_client", return_value=mock_client),
    ):
      blockers = json.loads(
          hlo_tools.query_hlo_graph(
              str(self.session_dir), mode="fusion_blockers"
          )
      )
      self.assertEqual(blockers["status"], "SUCCESS")
      self.assertEqual(blockers["fusion_blockers"][0]["blocker_name"], "bc")

      path_res = json.loads(
          hlo_tools.query_hlo_graph(
              str(self.session_dir),
              mode="shortest_path",
              src_op="x",
              dst_op="out",
          )
      )
      self.assertTrue(path_res["found"])
      self.assertEqual(path_res["distance"], 3)

      nb_up = hlo_tools.get_hlo_neighborhood(
          str(self.session_dir),
          instruction_name="fus",
          radius=1,
          direction="operands",
          detect_fusion_blockers=True,
          follow_calls=True,
      )
      self.assertIn("[FUSION_BLOCKER: bitcast]", nb_up)
      self.assertIn("sub_add", nb_up)
      self.assertNotIn("%out =", nb_up)

      transitions = json.loads(
          hlo_tools.query_hlo_graph(
              str(self.session_dir), mode="layout_transitions"
          )
      )
      self.assertEqual(transitions["status"], "SUCCESS")
      self.assertGreaterEqual(transitions["count"], 1)

      # Finding #1: fusion FLOPs are not double-counted in opcode_stats
      stats = json.loads(
          hlo_tools.query_hlo_graph(
              str(self.session_dir), mode="opcode_stats"
          )
      )
      total_flops = sum(r["total_flops"] for r in stats["rows"])
      self.assertAlmostEqual(total_flops, 512.0)

      # Finding #2: SQL guard rejects PRAGMA and write CTEs
      with self.assertRaises(ValueError):
        hlo_tools.query_hlo_graph(
            str(self.session_dir), mode="sql", sql="PRAGMA query_only(0)"
        )
      with self.assertRaises(ValueError):
        hlo_tools.query_hlo_graph(
            str(self.session_dir),
            mode="sql",
            sql="WITH x AS (SELECT 1) DELETE FROM instructions",
        )

      # Finding #3: invalid direction raises ValueError
      with self.assertRaises(ValueError):
        hlo_tools.get_hlo_neighborhood(
            str(self.session_dir), instruction_name="fus", direction="forward"
        )

      # Finding #4: max_nodes truncation appends notice
      nb_trunc = hlo_tools.get_hlo_neighborhood(
          str(self.session_dir), instruction_name="fus", radius=2, max_nodes=2
      )
      self.assertIn("[truncated at max_nodes=2;", nb_trunc)


class OssFetchDebugInfoTest(absltest.TestCase):
  """Tests for loading HloProto messages via the private descriptor pool."""

  # pylint: disable=protected-access

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.test_dir.cleanup)
    self.session_dir = pathlib.Path(self.test_dir.name)

  def _write_hlo_proto(self, filename: str) -> pathlib.Path:
    """Writes a small HloProto with a while loop and returns its path."""
    hlo = hlo_tools._hlo_proto_class()()
    module = hlo.hlo_module
    module.name = "jit_step"
    module.computations.add(name="body", id=1)
    module.computations.add(name="cond", id=2)
    entry = module.computations.add(name="main", id=3)
    module.entry_computation_id = 3
    param = entry.instructions.add(
        name="p0", opcode="parameter", id=10, parameter_number=0
    )
    param.shape.element_type = 11  # F32
    param.shape.dimensions.extend([2, 4])
    param.shape.layout.minor_to_major.extend([0, 1])
    entry.instructions.add(
        name="gte", opcode="get-tuple-element", id=11, tuple_index=1
    )
    entry.instructions.add(
        name="while.1", opcode="while", id=12, called_computation_ids=[1, 2]
    )
    path = self.session_dir / filename
    path.write_bytes(hlo.SerializeToString())
    return path

  def test_fetch_debug_info_parses_fields(self):
    path = self._write_hlo_proto("jit_step(123).hlo_proto.pb")
    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[path]
    ):
      debug_info = hlo_tools._fetch_debug_info("session")

    self.assertEqual(debug_info.program_id, [123])
    self.assertLen(debug_info.hlo_proto, 1)
    module = debug_info.hlo_proto[0].hlo_module
    self.assertEqual(module.name, "jit_step")
    instrs = {}
    for computation in module.computations:
      for instr in computation.instructions:
        instrs[instr.name] = instr
    self.assertEqual(instrs["p0"].parameter_number, 0)
    self.assertEqual(instrs["p0"].shape.element_type, 11)
    self.assertEqual(list(instrs["p0"].shape.dimensions), [2, 4])
    self.assertEqual(list(instrs["p0"].shape.layout.minor_to_major), [0, 1])
    self.assertEqual(instrs["gte"].tuple_index, 1)
    # While loops list [body, condition].
    self.assertEqual(list(instrs["while.1"].called_computation_ids), [1, 2])

  def test_parse_program_id(self):
    for name, expected in [
        ("jit_step(123).hlo_proto.pb", 123),
        ("jit_f(x)(42).hlo_proto.pb", 42),
        ("jit_step.hlo_proto.pb", 0),
        ("jit_step(abc).hlo_proto.pb", 0),
    ]:
      with self.subTest(name=name):
        self.assertEqual(
            hlo_tools._parse_program_id(pathlib.Path(name)), expected
        )

  def test_fetch_debug_info_no_files_raises(self):
    with mock.patch.object(hlo_tools, "_get_hlo_proto_files", return_value=[]):
      with self.assertRaises(FileNotFoundError):
        hlo_tools._fetch_debug_info("session")

  def test_fetch_debug_info_skips_corrupt_file(self):
    good = self._write_hlo_proto("jit_step(1).hlo_proto.pb")
    bad = self.session_dir / "jit_bad(2).hlo_proto.pb"
    bad.write_bytes(b"\xff\xff\xff")
    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[bad, good]
    ):
      debug_info = hlo_tools._fetch_debug_info("session")

    self.assertEqual(debug_info.program_id, [1])
    self.assertEqual(debug_info.hlo_proto[0].hlo_module.name, "jit_step")

  def test_fetch_debug_info_all_corrupt_raises(self):
    bad = self.session_dir / "jit_bad(2).hlo_proto.pb"
    bad.write_bytes(b"\xff\xff\xff")
    with mock.patch.object(
        hlo_tools, "_get_hlo_proto_files", return_value=[bad]
    ):
      with self.assertRaises(ValueError):
        hlo_tools._fetch_debug_info("session")

  def test_missing_descriptor_set_raises(self):
    hlo_tools._hlo_proto_class.cache_clear()
    self.addCleanup(hlo_tools._hlo_proto_class.cache_clear)
    with mock.patch.object(
        hlo_tools,
        "_HLO_DESCRIPTOR_SET_PATH",
        self.session_dir / "missing.pb",
    ):
      with self.assertRaises(FileNotFoundError):
        hlo_tools._hlo_proto_class()

  def test_hlo_proto_class_uses_private_pool(self):
    hlo_proto_class = hlo_tools._hlo_proto_class()
    self.assertEqual(hlo_proto_class.DESCRIPTOR.full_name, "xla.HloProto")
    self.assertIsNot(
        hlo_proto_class.DESCRIPTOR.file.pool, descriptor_pool.Default()
    )


if __name__ == "__main__":
  absltest.main()
