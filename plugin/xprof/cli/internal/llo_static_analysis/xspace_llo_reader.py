"""Reads `LloModuleProto`s out of an XSpace metadata plane.

The LLO debug pipeline serializes each lowered HLO instruction's LLO module as
an `XStat` `bytes_value` on every event of the `/host:metadata` plane, keyed by
the `llo_proto` stat metadata. This module walks that plane and yields the
parsed protos, mirroring the canonical C++ `ParseLloProtoFromXSpace` helper.

No producer or collector change is required: the full
`LloModuleProto` is already present on the served XSpace when LLO analysis was
requested at collection time.
"""

from collections.abc import Iterator
import io
import os

from absl import logging
from google.protobuf import message

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_module_pb2
from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import

# Mirrors the C++ XPlane schema (xplane_schema.cc).
_METADATA_PLANE_NAME = "/host:metadata"
_LLO_PROTO_STAT_NAME = "llo_proto"


def load_xspace(path: str) -> xplane_pb2.XSpace:
  """Loads a binary-serialized `XSpace` proto from `path`."""
  if not os.path.exists(path):
    raise FileNotFoundError(f"xspace file not found: {path}")
  with io.open(path, "rb") as f:
    raw = f.read()
  xspace = xplane_pb2.XSpace()
  xspace.ParseFromString(raw)
  return xspace


def iter_llo_modules(
    xspace: xplane_pb2.XSpace,
) -> Iterator[llo_module_pb2.LloModuleProto]:
  """Yields every `LloModuleProto` carried by `xspace`'s metadata plane.

  Each event metadata in the `/host:metadata` plane carries one serialized
  `LloModuleProto` in an `XStat` keyed by the plane-local `llo_proto` stat
  metadata id. Malformed stats are skipped rather than raising.

  Args:
    xspace: The `XSpace` proto whose `/host:metadata` plane is inspected.
  """
  for plane in xspace.planes:
    if plane.name != _METADATA_PLANE_NAME:
      continue
    llo_stat_metadata_id = None
    for sm_id, sm in plane.stat_metadata.items():
      if sm.name == _LLO_PROTO_STAT_NAME:
        llo_stat_metadata_id = sm_id
        break
    if llo_stat_metadata_id is None:
      continue
    for em in plane.event_metadata.values():
      for stat in em.stats:
        if stat.metadata_id != llo_stat_metadata_id:
          continue
        if not stat.HasField("bytes_value"):
          continue
        module = llo_module_pb2.LloModuleProto()
        try:
          module.ParseFromString(stat.bytes_value)
        except message.DecodeError:
          logging.warning("Skipping unparseable LLO payload on %r", em.name)
          continue
        yield module


def find_llo_modules_for_op(
    xspace: xplane_pb2.XSpace, hlo_op: str
) -> list[llo_module_pb2.LloModuleProto]:
  """Returns every LLO module whose `hlo_instruction_name` contains `hlo_op`.

  Match is case-sensitive substring. One HLO op may lower to several LLO
  modules (e.g. overlay split / multi-stream), so a list is returned.

  Args:
    xspace: The `XSpace` proto to search.
    hlo_op: Substring matched against `hlo_instruction_name`.
  """
  return [
      m for m in iter_llo_modules(xspace) if hlo_op in m.hlo_instruction_name
  ]


def list_hlo_op_names(xspace: xplane_pb2.XSpace) -> list[str]:
  """Returns the sorted unique `hlo_instruction_name`s across LLO modules."""
  seen: set[str] = set()
  for m in iter_llo_modules(xspace):
    seen.add(m.hlo_instruction_name)
  return sorted(seen)
