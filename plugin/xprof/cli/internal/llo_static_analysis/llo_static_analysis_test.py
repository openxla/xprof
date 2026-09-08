"""Hermetic tests for the llo_static_analysis foundational library.

Builds an in-memory `LloModuleProto` + `XSpace` that mirror what the C++
`ParseLloProtoFromXSpace` produces on a real trace, so the LLO analyses can be
validated without a TPU run.
"""

import json
import os

from absl.testing import absltest

from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import

from xprof.embedded.llo_analysis import llo_lite_pb2
from xprof.cli.internal.llo_static_analysis import llo_bundle_utilization
from xprof.cli.internal.llo_static_analysis import llo_region_tree
from xprof.cli.internal.llo_static_analysis import llo_source_map
from xprof.cli.internal.llo_static_analysis import xspace_llo_reader

_METADATA_PLANE_NAME = "/host:metadata"
_LLO_PROTO_STAT_NAME = "llo_proto"


def _make_static_util(num_bundles=30):
  util = llo_lite_pb2.StaticPerBundleUtilizationProto()
  util.vector_alu.denominator = 2
  util.vector_alu.numerator.extend([(i % 2) for i in range(num_bundles)])
  util.mxu.denominator = 1
  util.mxu.numerator.extend([1 if i == 5 else 0 for i in range(num_bundles)])
  util.vector_spill.extend([0] * num_bundles)
  util.vector_fill.extend([0] * num_bundles)
  return util


def _add_instruction(
    region, ordinal, bundle, reg_id=-1, reg_type=None, operand_ordinals=()
):
  inst = region.members.add().instruction
  inst.ordinal = ordinal
  inst.scheduled_bundleno = bundle
  if reg_id >= 0:
    inst.register_id = reg_id
  if reg_type is not None:
    inst.register_type = reg_type
  for op_ord in operand_ordinals:
    inst.operands.add().instruction_ordinal = op_ord
  return inst


def _make_module(name="kScan", with_loop=True, with_static_util=True):
  m = llo_lite_pb2.LloModuleProto()
  m.hlo_instruction_name = name
  m.hlo_module_name = "TestModule"
  m.hlo_module_id = 42

  top = m.top_region
  top.name = "top"
  top.start_bundleno = 0
  top.limit_bundleno = 30
  top.ordinal = 1

  prologue = top.members.add().sub_region
  prologue.name = "prologue"
  prologue.start_bundleno = 0
  prologue.limit_bundleno = 5
  prologue.ordinal = 2
  # Instruction at bundle 2, defines VREG 10.
  _add_instruction(
      prologue,
      ordinal=100,
      bundle=2,
      reg_id=10,
      reg_type=llo_lite_pb2.REGISTER_TYPE_VREG,
  )

  if with_loop:
    loop_member = top.members.add().loop
    loop_member.kind = llo_lite_pb2.LloLoopProto.LOOP_KIND_WHILE
    loop_member.index_space.start = 0
    loop_member.index_space.limit = 2
    loop_member.index_space.step = 1

    pre = loop_member.pre_header
    pre.name = "loop_pre_header"
    pre.start_bundleno = 5
    pre.limit_bundleno = 7
    pre.ordinal = 4

    hdr = loop_member.header
    hdr.name = "loop_header"
    hdr.start_bundleno = 7
    hdr.limit_bundleno = 9
    hdr.ordinal = 5

    body = loop_member.body
    body.name = "loop_body"
    body.start_bundleno = 9
    body.limit_bundleno = 19
    body.ordinal = 6
    # Producer at bundle 10 (VREG 20), consumer at bundle 12 uses it.
    _add_instruction(
        body,
        ordinal=200,
        bundle=10,
        reg_id=20,
        reg_type=llo_lite_pb2.REGISTER_TYPE_VREG,
    )
    _add_instruction(
        body,
        ordinal=201,
        bundle=12,
        reg_id=21,
        reg_type=llo_lite_pb2.REGISTER_TYPE_VREG,
        operand_ordinals=(200,),
    )

    footer = loop_member.footer
    footer.name = "loop_footer"
    footer.start_bundleno = 19
    footer.limit_bundleno = 21
    footer.ordinal = 7

    exit_r = loop_member.exit
    exit_r.name = "loop_exit"
    exit_r.start_bundleno = 21
    exit_r.limit_bundleno = 25
    exit_r.ordinal = 8

  epilogue = top.members.add().sub_region
  epilogue.name = "epilogue"
  epilogue.start_bundleno = 25
  epilogue.limit_bundleno = 30
  epilogue.ordinal = 9

  if with_static_util:
    m.static_utilization.CopyFrom(_make_static_util(num_bundles=30))

  # Source map: one location for instruction ordinal 200 (bundle 10).
  sm = m.source_map
  sm.strings.append("scan_emitter.cc")
  loc = sm.locations.add()
  frame = loc.frames.add()
  frame.path = 0
  frame.line_start = 1837
  loc.ordinals.append(200)

  return m


def _make_xspace_with_module(module):
  xspace = xplane_pb2.XSpace()
  plane = xspace.planes.add()
  plane.name = _METADATA_PLANE_NAME
  plane.stat_metadata[1].id = 1
  plane.stat_metadata[1].name = _LLO_PROTO_STAT_NAME
  em = plane.event_metadata[1]
  em.id = 1
  em.name = (
      f"{module.hlo_module_name}({module.hlo_module_id})"
      f"::{module.hlo_instruction_name}"
  )
  stat = em.stats.add()
  stat.metadata_id = 1
  stat.bytes_value = module.SerializeToString()
  return xspace


class LloStaticAnalysisTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.module = _make_module(name="kScan")
    self.xspace = _make_xspace_with_module(self.module)
    self.tmpdir = self.create_tempdir().full_path
    self.xspace_path = os.path.join(self.tmpdir, "test.pb")
    with open(self.xspace_path, "wb") as f:
      f.write(self.xspace.SerializeToString())

  # ---- reader ----
  def test_load_xspace_roundtrip(self):
    loaded = xspace_llo_reader.load_xspace(self.xspace_path)
    self.assertLen(loaded.planes, 1)

  def test_iter_llo_modules_finds_embedded(self):
    loaded = xspace_llo_reader.load_xspace(self.xspace_path)
    modules = list(xspace_llo_reader.iter_llo_modules(loaded))
    self.assertLen(modules, 1)
    self.assertEqual(modules[0].hlo_instruction_name, "kScan")

  def test_list_hlo_op_names(self):
    loaded = xspace_llo_reader.load_xspace(self.xspace_path)
    self.assertEqual(xspace_llo_reader.list_hlo_op_names(loaded), ["kScan"])

  def test_find_llo_modules_for_op(self):
    loaded = xspace_llo_reader.load_xspace(self.xspace_path)
    self.assertLen(
        xspace_llo_reader.find_llo_modules_for_op(loaded, "kScan"), 1
    )
    self.assertEmpty(xspace_llo_reader.find_llo_modules_for_op(loaded, "kNope"))

  def test_reader_missing_file(self):
    with self.assertRaises(FileNotFoundError):
      xspace_llo_reader.load_xspace("/nonexistent/path.pb")

  # ---- region tree ----
  def test_llo_tree_renders_json(self):
    raw = llo_region_tree.render_tree_json(self.module)
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kScan")
    self.assertEqual(res["total_bundles"], 30)
    self.assertEqual(res["root"]["name"], "top")
    child_names = [c["name"] for c in res["root"]["children"]]
    self.assertIn("prologue", child_names)
    self.assertIn("loop_body", child_names)
    self.assertIn("epilogue", child_names)
    loop_node = [
        c for c in res["root"]["children"] if c["name"] == "loop_body"
    ][0]
    self.assertEqual(loop_node["trip_count"], 2)
    self.assertEqual(loop_node["step"], 1)

  def test_llo_tree_no_loop(self):
    m = _make_module(with_loop=False)
    raw = llo_region_tree.render_tree_json(m)
    res = json.loads(raw)
    for c in res["root"]["children"]:
      self.assertNotIn("trip_count", c)

  def test_bundle_to_region_path_attribution(self):
    paths = llo_region_tree.build_bundle_to_region_path(self.module)
    self.assertIn("prologue", paths.get(2, ""))
    self.assertIn("loop_body", paths.get(10, ""))
    self.assertIn("epilogue", paths.get(27, ""))

  def test_iter_instructions(self):
    insts = list(llo_region_tree.iter_instructions(self.module))
    ordinals = sorted(inst.ordinal for _, inst in insts)
    self.assertEqual(ordinals, [100, 200, 201])

  # ---- bundle utilization ----
  def test_bundle_util_renders_json(self):
    raw = llo_bundle_utilization.render_bundle_util_json(
        self.module, max_rows=10
    )
    res = json.loads(raw)
    self.assertEqual(res["hlo_instruction_name"], "kScan")
    self.assertEqual(res["total_bundles"], 30)
    self.assertLen(res["bundles"], 10)
    self.assertEqual(res["bundles"][5]["mxu_used"], 1)
    self.assertEqual(res["bundles"][5]["util_pct"], 67)

  def test_bundle_util_structured(self):
    recs = llo_bundle_utilization.extract_bundle_utilization(self.module)
    self.assertLen(recs, 30)
    # Bundle 5: MXU 1/1 + V_ALU 1/2 -> used 2 of 3 -> 67%.
    self.assertEqual(recs[5]["mxu_used"], 1)
    self.assertEqual(recs[5]["util_pct"], 67)

  def test_bundle_util_empty(self):
    m = _make_module(with_static_util=False)
    self.assertFalse(llo_bundle_utilization.has_static_utilization(m))
    raw = llo_bundle_utilization.render_bundle_util_json(m)
    res = json.loads(raw)
    self.assertEqual(res["total_bundles"], 0)
    self.assertEmpty(res["bundles"])
    self.assertEmpty(llo_bundle_utilization.extract_bundle_utilization(m))

  def test_bundle_util_with_region_path(self):
    paths = llo_region_tree.build_bundle_to_region_path(self.module)
    raw = llo_bundle_utilization.render_bundle_util_json(
        self.module, max_rows=10, region_path_map=paths
    )
    res = json.loads(raw)
    self.assertIn("prologue", res["bundles"][2]["region_path"])

  # ---- source map (exact join) ----
  def test_ordinal_to_bundle(self):
    mapping = llo_source_map.build_ordinal_to_bundle(self.module)
    self.assertEqual(mapping, {100: 2, 200: 10, 201: 12})

  def test_source_by_bundle_is_exact(self):
    by_bundle = llo_source_map.build_source_by_bundle(self.module)
    # Instruction 200 issues at bundle 10 -> exact mapping to bundle 10 only.
    self.assertIn("scan_emitter.cc:1837", by_bundle.get(10, []))
    # NOT smeared across the whole loop_body region [9, 19).
    self.assertNotIn(9, by_bundle)
    self.assertNotIn(11, by_bundle)

  def test_source_for_bundle(self):
    raw = llo_source_map.render_source_for_bundle(self.module, 10)
    res = json.loads(raw)
    self.assertIn("scan_emitter.cc:1837", res["sources"])

  def test_source_for_bundle_no_match(self):
    raw = llo_source_map.render_source_for_bundle(self.module, 28)
    res = json.loads(raw)
    self.assertEmpty(res["sources"])

  def test_source_inverse(self):
    raw = llo_source_map.render_bundles_for_source(
        self.module, "scan_emitter.cc"
    )
    res = json.loads(raw)
    self.assertLen(res["matches"], 1)
    self.assertEqual(res["matches"][0]["source"], "scan_emitter.cc:1837")
    self.assertEqual(res["matches"][0]["bundles"], [10])

  def test_source_region_fallback(self):
    # A location that references a region ordinal (9 = epilogue), not a
    # scheduled instruction, should fall back to the region bundle range.
    m = _make_module()
    loc = m.source_map.locations.add()
    fr = loc.frames.add()
    fr.path = 0
    fr.line_start = 4242
    loc.ordinals.append(9)  # epilogue region ordinal, no such instruction.
    by_bundle = llo_source_map.build_source_by_bundle(m)
    self.assertIn("scan_emitter.cc:4242", by_bundle.get(25, []))
    self.assertIn("scan_emitter.cc:4242", by_bundle.get(29, []))

  def test_source_region_fallback_clamps_huge_and_negative_ranges(self):
    m = _make_module()
    bad_neg = m.top_region.members.add().sub_region
    bad_neg.name = "bad_neg"
    bad_neg.ordinal = 90
    bad_neg.start_bundleno = -10
    bad_neg.limit_bundleno = 5

    huge = m.top_region.members.add().sub_region
    huge.name = "huge"
    huge.ordinal = 91
    huge.start_bundleno = 0
    huge.limit_bundleno = 2_147_483_647

    loc = m.source_map.locations.add()
    fr = loc.frames.add()
    fr.path = 0
    fr.line_start = 9999
    loc.ordinals.extend([90, 91])

    by_bundle = llo_source_map.build_source_by_bundle(m)
    self.assertNotIn(-10, by_bundle)
    self.assertLessEqual(
        len(by_bundle), llo_source_map.MAX_BUNDLE_RANGE_SPAN + 10
    )


if __name__ == "__main__":
  absltest.main()
