"""Target/hardware argument surfacing for an LLO module (capability K).

`LloModuleProto.target_arguments` (an `xla.TargetArgumentsProto`) is
proto-resident, so the *raw* target arguments the compiler recorded are
available directly on the proto. This module surfaces them as JSON.

Residual gap (documented, not silently dropped): the authoritative per-core
hardware *constants* -- mxus/xlus per tensor core, VMEM/SMEM/CMEM
sizes, tensor-core / HBM frequencies, HBM capacity -- are NOT stored in this
proto. They are computed by an out-of-tree target catalog keyed by the TPU
version + `chip_config_name`. Reproducing those exact constants from Python
would require a dependency on that out-of-tree target catalog. Until
then, this mode exposes what the proto carries (TPU version, variant, platform
type, chip config name, chip/host dimensions, replica count, reserved HBM) and
flags the constants that are only derivable via the catalog.
"""

import json

from xprof.protobuf import llo_lite_pb2


def _fmt_dims(dims) -> str:
  """Formats a TpuDimensionsProto as an x/y/z[/w] tuple."""
  parts = [f"x={dims.x}", f"y={dims.y}", f"z={dims.z}"]
  try:
    if dims.HasField("w"):
      parts.append(f"w={dims.w}")
  except ValueError:
    pass
  return ", ".join(parts)


def extract_key_fields(
    module: llo_lite_pb2.LloModuleProto,
) -> dict[str, object]:
  """Returns the proto-resident target fields as a flat dict (best-effort)."""
  out: dict[str, object] = {}
  if not module.HasField("target_arguments"):
    return out
  ta = module.target_arguments
  out["replica_count"] = ta.replica_count
  out["reserved_hbm_usage_bytes"] = ta.reserved_hbm_usage_bytes
  if ta.HasField("tpu_topology_args"):
    topo = ta.tpu_topology_args
    out["variant"] = topo.variant
    out["chip_config_name"] = topo.chip_config_name
    out["platform_type"] = topo.platform_type
    out["twist"] = topo.twist
    out["enhanced_barrier_enabled"] = topo.enhanced_barrier_enabled
    if topo.HasField("chips_per_host_bounds"):
      out["chips_per_host_bounds"] = _fmt_dims(topo.chips_per_host_bounds)
    if topo.HasField("host_bounds"):
      out["host_bounds"] = _fmt_dims(topo.host_bounds)
    if topo.HasField("version"):
      out["version"] = str(topo.version).strip().replace("\n", " ")
  return out


def render_target_info_json(
    module: llo_lite_pb2.LloModuleProto,
) -> str:
  """Renders proto-resident target arguments + the documented catalog gap as JSON."""
  has_target = module.HasField("target_arguments")
  fields = extract_key_fields(module) if has_target else {}
  raw_proto = (
      str(module.target_arguments).rstrip() or "(empty)" if has_target else ""
  )
  payload = {
      "hlo_instruction_name": module.hlo_instruction_name,
      "has_target_arguments": has_target,
      "key_fields": fields,
      "raw_target_arguments": raw_proto,
      "catalog_only_fields": [
          "mxus / xlus per tensor core",
          "VMEM / SMEM / CMEM sizes",
          "tensor-core and HBM frequencies",
          "HBM capacity per chip",
          "tensor cores / sparse cores per chip",
      ],
  }
  return json.dumps(payload, indent=2)
