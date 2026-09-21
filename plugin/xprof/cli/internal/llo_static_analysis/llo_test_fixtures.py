"""Test-only fixtures that build synthetic LLO XSpaces.

Centralizes construction of `LloModuleProto` + `XSpace` fixtures using
`llo_lite_pb2` so tests in other packages can exercise the LLO analyses
hermetically.
"""

import os
from typing import Any

from xprof.protobuf import llo_lite_pb2

try:
  from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

  _XSpaceProto = xplane_pb2.XSpace
except ImportError:
  _XSpaceProto = llo_lite_pb2.LloXSpaceProto

METADATA_PLANE_NAME = "/host:metadata"
LLO_PROTO_STAT_NAME = "llo_proto"


def make_sample_module(name: str = "kScan") -> llo_lite_pb2.LloModuleProto:
  """Builds a small but representative LLO module.

  Contains a top region with a scheduled VREG-producing instruction at bundle 3,
  a spill pseudo at bundle 4, static utilization (MXU busy at bundle 3, one
  spill at bundle 4), and a source-map entry linking ordinal 100 to
  `emitter.cc:99`.

  Args:
    name: The `hlo_instruction_name` assigned to the synthetic module.

  Returns:
    A populated `LloModuleProto` test fixture.
  """
  m = llo_lite_pb2.LloModuleProto()
  m.hlo_instruction_name = name
  m.hlo_module_name = "TestModule"
  m.hlo_module_id = 7
  top = m.top_region
  top.name = "top"
  top.start_bundleno = 0
  top.limit_bundleno = 6
  top.ordinal = 1

  inst = top.members.add().instruction
  inst.ordinal = 100
  inst.opcode = llo_lite_pb2.OPCODE_VECTOR_MATMUL
  inst.scheduled_bundleno = 3
  inst.register_id = 1
  inst.register_type = llo_lite_pb2.REGISTER_TYPE_VREG
  # Attach a compiler annotation (interned) exercising BDI stall codes.
  inst.annotation_handle = len(m.interned_strings)
  m.interned_strings.append("bdi:R:2,O:1 scope:VMEM")

  spill = top.members.add().instruction
  spill.ordinal = 101
  spill.scheduled_bundleno = 4
  spill.pseudo_kind = llo_lite_pb2.PSEUDO_KIND_SPILL_TO_MEMORY

  m.static_utilization.mxu.denominator = 1
  m.static_utilization.mxu.numerator.extend([0, 0, 0, 1, 0, 0])
  m.static_utilization.vector_spill.extend([0, 0, 0, 0, 1, 0])
  m.static_utilization.vector_fill.extend([0] * 6)

  # Proto-resident target arguments (capability K).
  ta = m.target_arguments
  ta.replica_count = 4
  ta.reserved_hbm_usage_bytes = 1048576
  ta.tpu_topology_args.chip_config_name = "df"
  ta.tpu_topology_args.variant = "pf"

  sm = m.source_map
  sm.strings.append("emitter.cc")
  loc = sm.locations.add()
  fr = loc.frames.add()
  fr.path = 0
  fr.line_start = 99
  loc.ordinals.append(100)
  return m


def make_xspace_with_module(
    module: llo_lite_pb2.LloModuleProto,
) -> Any:
  """Wraps `module` into an XSpace metadata plane."""
  xspace = _XSpaceProto()
  plane = xspace.planes.add()
  plane.name = METADATA_PLANE_NAME
  plane.stat_metadata[1].id = 1
  plane.stat_metadata[1].name = LLO_PROTO_STAT_NAME
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


def write_sample_xspace(tmpdir: str, name: str = "kScan") -> str:
  """Writes a sample XSpace into `tmpdir` and returns its path.

  Args:
    tmpdir: Destination directory. Pass `self.create_tempdir().full_path` so the
      test framework owns cleanup; this function deliberately does not create a
      temp directory itself, which would leak one per test.
    name: `hlo_instruction_name` for the synthetic module.

  Returns:
    The path of the serialized XSpace.
  """
  xspace = make_xspace_with_module(make_sample_module(name))
  path = os.path.join(tmpdir, "xspace.pb")
  with open(path, "wb") as f:
    f.write(xspace.SerializeToString())
  return path
