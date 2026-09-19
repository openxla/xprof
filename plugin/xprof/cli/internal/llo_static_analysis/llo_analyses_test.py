"""Hermetic tests for register-pressure, spill/allocation, opcode, BDI, and target analyses."""

import json

from absl.testing import absltest

from xprof.embedded.llo_analysis import llo_lite_pb2
from xprof.cli.internal.llo_static_analysis import bdi_stalls
from xprof.cli.internal.llo_static_analysis import llo_allocations
from xprof.cli.internal.llo_static_analysis import llo_opcode_stats
from xprof.cli.internal.llo_static_analysis import register_pressure
from xprof.cli.internal.llo_static_analysis import target_info

_VREG = llo_lite_pb2.REGISTER_TYPE_VREG
_SREG = llo_lite_pb2.REGISTER_TYPE_SREG


def _add_inst(
    region, ordinal, bundle, reg_id=-1, reg_type=None, operand_ordinals=()
):
  inst = region.members.add().instruction
  inst.ordinal = ordinal
  inst.scheduled_bundleno = bundle
  if reg_id >= 0:
    inst.register_id = reg_id
  if reg_type is not None:
    inst.register_type = reg_type
  for op in operand_ordinals:
    inst.operands.add().instruction_ordinal = op
  return inst


def _make_module():
  m = llo_lite_pb2.LloModuleProto()
  m.hlo_instruction_name = "kFusion"
  top = m.top_region
  top.name = "top"
  top.start_bundleno = 0
  top.limit_bundleno = 20
  top.ordinal = 1

  # Producers.
  _add_inst(top, ordinal=1, bundle=0, reg_id=1, reg_type=_VREG)
  _add_inst(top, ordinal=2, bundle=1, reg_id=2, reg_type=_VREG)
  _add_inst(top, ordinal=3, bundle=2, reg_id=3, reg_type=_SREG)
  # Consumers (no register produced).
  _add_inst(top, ordinal=10, bundle=5, operand_ordinals=(1,))
  _add_inst(top, ordinal=11, bundle=6, operand_ordinals=(2,))
  _add_inst(top, ordinal=12, bundle=3, operand_ordinals=(3,))

  # A spill pseudo-instruction at bundle 4.
  spill = _add_inst(top, ordinal=20, bundle=4)
  spill.pseudo_kind = llo_lite_pb2.PSEUDO_KIND_SPILL_TO_MEMORY

  # Interned allocations.
  a1 = m.interned_constants.add().allocation
  a1.ordinal = 50
  a1.size = 1024
  a1.is_spill = True
  a2 = m.interned_constants.add().allocation
  a2.ordinal = 51
  a2.size = 512
  a2.is_scoped = True
  # A non-allocation constant must be ignored.
  m.interned_constants.add().bool_value = True

  # Static spill/fill counts.
  m.static_utilization.vector_spill.extend([0, 0, 1, 0, 2])
  m.static_utilization.vector_fill.extend([0, 1, 0, 0, 0])
  return m


class RegisterPressureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.module = _make_module()

  def test_live_ranges(self):
    ranges = {
        r.ordinal: (r.start, r.end)
        for r in register_pressure.compute_live_ranges(self.module)
    }
    self.assertEqual(ranges[1], (0, 5))
    self.assertEqual(ranges[2], (1, 6))
    self.assertEqual(ranges[3], (2, 3))
    # Consumers produce no register -> not in ranges.
    self.assertNotIn(10, ranges)

  def test_peak_pressure(self):
    peaks = register_pressure.compute_peak_pressure(self.module)
    # VREG 1 [0,5] and VREG 2 [1,6] overlap -> peak 2.
    self.assertEqual(peaks[_VREG], 2)
    self.assertEqual(peaks[_SREG], 1)

  def test_pressure_series(self):
    series = register_pressure.compute_pressure_series(self.module)
    # Bundle 2: VREG {1,2} live = 2, SREG {3} live = 1.
    self.assertEqual(series[2][_VREG], 2)
    self.assertEqual(series[2][_SREG], 1)

  def test_render_json(self):
    raw = register_pressure.render_register_pressure_json(self.module)
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kFusion")
    self.assertEqual(res["peak_by_register_type"]["VREG"], 2)
    self.assertTrue(res["hottest_bundles"])

  def test_render_empty(self):
    empty = llo_lite_pb2.LloModuleProto()
    empty.hlo_instruction_name = "kEmpty"
    raw = register_pressure.render_register_pressure_json(empty)
    res = json.loads(raw)
    self.assertEmpty(res["peak_by_register_type"])

  def test_huge_bundle_span_bounded(self):
    m = llo_lite_pb2.LloModuleProto()
    m.hlo_instruction_name = "kHuge"
    top = m.top_region
    _add_inst(top, ordinal=1, bundle=0, reg_id=1, reg_type=_VREG)
    _add_inst(
        top,
        ordinal=2,
        bundle=1_000_000_000,
        reg_id=2,
        reg_type=_VREG,
        operand_ordinals=(1,),
    )
    peaks = register_pressure.compute_peak_pressure(m)
    self.assertEqual(peaks[_VREG], 2)
    series = register_pressure.compute_pressure_series(m)
    self.assertLessEqual(
        len(series), register_pressure.MAX_BUNDLE_RANGE_SPAN + 5
    )
    raw = register_pressure.render_register_pressure_json(m, top_n=-5)
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kHuge")


class AllocationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.module = _make_module()

  def test_extract_allocations(self):
    allocs = llo_allocations.extract_allocations(self.module)
    # Two allocations; the bool constant is filtered out.
    self.assertLen(allocs, 2)
    self.assertTrue(allocs[0]["is_spill"])

  def test_spill_fill_instructions(self):
    recs = llo_allocations.extract_spill_fill_instructions(self.module)
    self.assertLen(recs, 1)
    self.assertEqual(recs[0]["kind"], "SPILL_TO_MEMORY")
    self.assertEqual(recs[0]["bundle"], 4)

  def test_static_totals(self):
    totals = llo_allocations.static_spill_fill_totals(self.module)
    self.assertEqual(totals["spill"], 3)
    self.assertEqual(totals["fill"], 1)

  def test_summarize(self):
    s = llo_allocations.summarize(self.module)
    self.assertEqual(s["num_allocations"], 2)
    self.assertEqual(s["num_spill_allocations"], 1)
    self.assertEqual(s["num_scoped_allocations"], 1)
    self.assertEqual(s["total_allocation_bytes"], 1536)
    self.assertEqual(s["num_spill_fill_instructions"], 1)

  def test_render_json(self):
    raw = llo_allocations.render_allocations_json(self.module)
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kFusion")
    self.assertEqual(
        res["spill_fill_instructions"][0]["kind"], "SPILL_TO_MEMORY"
    )

  def test_empty_module(self):
    empty = llo_lite_pb2.LloModuleProto()
    self.assertEmpty(llo_allocations.extract_allocations(empty))
    self.assertEqual(
        llo_allocations.static_spill_fill_totals(empty),
        {"spill": 0, "fill": 0},
    )


class OpcodeStatsTest(absltest.TestCase):

  def _make_opcode_module(self):
    m = llo_lite_pb2.LloModuleProto()
    m.hlo_instruction_name = "kOpcodes"
    top = m.top_region
    top.name = "top"
    top.start_bundleno = 0
    top.limit_bundleno = 10
    top.ordinal = 1
    # Two matmuls (named OPCODE_VECTOR_MATMUL* -> matrix, not vector), one
    # scalar add, one transpose (crosslane), one DMA.
    _add_inst(top, ordinal=1, bundle=0).opcode = (
        llo_lite_pb2.OPCODE_VECTOR_MATMUL
    )
    _add_inst(top, ordinal=2, bundle=1).opcode = (
        llo_lite_pb2.OPCODE_VECTOR_MATMUL
    )
    _add_inst(top, ordinal=3, bundle=2).opcode = (
        llo_lite_pb2.OPCODE_SCALAR_SELECT
    )
    _add_inst(top, ordinal=4, bundle=3).opcode = (
        llo_lite_pb2.OPCODE_VECTOR_TRANSPOSE
    )
    _add_inst(top, ordinal=5, bundle=4).opcode = (
        llo_lite_pb2.OPCODE_DMA_HBM_TO_VMEM
    )
    return m

  def test_opcode_name_roundtrip(self):
    self.assertEqual(
        llo_opcode_stats.opcode_name(llo_lite_pb2.OPCODE_VECTOR_MATMUL),
        "OPCODE_VECTOR_MATMUL",
    )
    self.assertEqual(llo_opcode_stats.opcode_name(-999), "OPCODE_-999")

  def test_matmul_classified_as_matrix_not_vector(self):
    # Regression guard: OPCODE_VECTOR_MATMUL is a matrix-unit op despite the
    # "VECTOR" prefix.
    self.assertEqual(
        llo_opcode_stats.coarse_category("OPCODE_VECTOR_MATMUL"), "matrix"
    )
    self.assertEqual(
        llo_opcode_stats.coarse_category("OPCODE_VECTOR_TRANSPOSE"), "crosslane"
    )
    self.assertEqual(
        llo_opcode_stats.coarse_category("OPCODE_VECTOR_SELECT"), "vector"
    )
    self.assertEqual(
        llo_opcode_stats.coarse_category("OPCODE_DMA_HBM_TO_VMEM"), "dma"
    )

  def test_histogram(self):
    m = self._make_opcode_module()
    hist = llo_opcode_stats.opcode_histogram(m)
    self.assertEqual(hist["OPCODE_VECTOR_MATMUL"], 2)
    self.assertEqual(hist["OPCODE_SCALAR_SELECT"], 1)

  def test_category_histogram(self):
    m = self._make_opcode_module()
    cats = llo_opcode_stats.category_histogram(m)
    self.assertEqual(cats["matrix"], 2)
    self.assertEqual(cats["scalar"], 1)
    self.assertEqual(cats["crosslane"], 1)
    self.assertEqual(cats["dma"], 1)

  def test_render_json(self):
    raw = llo_opcode_stats.render_opcode_stats_json(self._make_opcode_module())
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kOpcodes")
    self.assertEqual(res["top_opcodes"][0]["opcode"], "OPCODE_VECTOR_MATMUL")
    self.assertEqual(res["categories"][0]["category"], "matrix")

  def test_render_empty(self):
    empty = llo_lite_pb2.LloModuleProto()
    empty.hlo_instruction_name = "kEmpty"
    raw = llo_opcode_stats.render_opcode_stats_json(empty)
    res = json.loads(raw)
    self.assertEqual(res["total_instructions"], 0)


class BdiStallsTest(absltest.TestCase):

  def _make_module(self):
    m = llo_lite_pb2.LloModuleProto()
    m.hlo_instruction_name = "kBdi"
    top = m.top_region
    top.name = "top"
    top.start_bundleno = 0
    top.limit_bundleno = 10
    top.ordinal = 1
    # interned_strings[0] carries two BDI codes (R, O) plus a scope suffix;
    # index 1 is a non-BDI annotation.
    m.interned_strings.append("bdi:R:3,O:1,Z:9 scope:VMEM")
    m.interned_strings.append("scope:SMEM")
    i0 = _add_inst(top, ordinal=1, bundle=0)
    i0.annotation_handle = 0
    i1 = _add_inst(top, ordinal=2, bundle=1)
    i1.annotation_handle = 1  # no bdi codes
    _add_inst(top, ordinal=3, bundle=2)  # no annotation at all
    return m

  def test_parse_bdi_codes(self):
    self.assertEqual(
        bdi_stalls.parse_bdi_codes("bdi:R:3,O:1,Z:9 scope:VMEM"), ["R", "O"]
    )
    # Unknown letters (Z) and non-key parts are ignored; duplicates collapse.
    self.assertEqual(bdi_stalls.parse_bdi_codes("bdi:R:1,R:2"), ["R"])
    self.assertEqual(bdi_stalls.parse_bdi_codes("no marker here"), [])
    self.assertEqual(bdi_stalls.parse_bdi_codes(None), [])

  def test_annotation_for(self):
    m = self._make_module()
    insts = [mem.instruction for mem in m.top_region.members]
    self.assertEqual(
        bdi_stalls.annotation_for(m, insts[0]), "bdi:R:3,O:1,Z:9 scope:VMEM"
    )
    self.assertIsNone(bdi_stalls.annotation_for(m, insts[2]))

  def test_iter_and_histogram(self):
    m = self._make_module()
    recs = bdi_stalls.iter_bdi_instructions(m)
    self.assertLen(recs, 1)
    self.assertEqual(recs[0], (1, 0, ["R", "O"]))
    self.assertEqual(bdi_stalls.bdi_code_histogram(m), {"R": 1, "O": 1})

  def test_render_json(self):
    raw = bdi_stalls.render_bdi_stalls_json(self._make_module())
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kBdi")
    self.assertEqual(res["code_histogram"]["R"], 1)

  def test_render_empty(self):
    empty = llo_lite_pb2.LloModuleProto()
    empty.hlo_instruction_name = "kEmpty"
    raw = bdi_stalls.render_bdi_stalls_json(empty)
    res = json.loads(raw)
    self.assertEqual(res["total_annotated_instructions"], 0)
    self.assertEmpty(res["code_histogram"])


class TargetInfoTest(absltest.TestCase):

  def _make_module(self):
    m = llo_lite_pb2.LloModuleProto()
    m.hlo_instruction_name = "kTarget"
    ta = m.target_arguments
    ta.replica_count = 8
    ta.reserved_hbm_usage_bytes = 2048
    ta.tpu_topology_args.chip_config_name = "df"
    ta.tpu_topology_args.variant = "pf"
    return m

  def test_extract_key_fields(self):
    fields = target_info.extract_key_fields(self._make_module())
    self.assertEqual(fields["replica_count"], 8)
    self.assertEqual(fields["chip_config_name"], "df")
    self.assertEqual(fields["variant"], "pf")

  def test_extract_key_fields_empty(self):
    empty = llo_lite_pb2.LloModuleProto()
    self.assertEmpty(target_info.extract_key_fields(empty))

  def test_render_json(self):
    raw = target_info.render_target_info_json(self._make_module())
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kTarget")
    self.assertEqual(res["key_fields"]["chip_config_name"], "df")
    # The documented catalog gap must be surfaced, not silently dropped.
    self.assertIn("mxus / xlus per tensor core", res["catalog_only_fields"])

  def test_render_absent(self):
    empty = llo_lite_pb2.LloModuleProto()
    empty.hlo_instruction_name = "kEmpty"
    raw = target_info.render_target_info_json(empty)
    res = json.loads(raw)
    self.assertFalse(res["has_target_arguments"])
    self.assertEmpty(res["key_fields"])


if __name__ == "__main__":
  absltest.main()
