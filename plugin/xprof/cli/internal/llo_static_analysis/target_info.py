"""Target/hardware argument surfacing for an LLO module (capability K).

`LloModuleProto.target_arguments` (an `xla.TargetArgumentsProto`) is
proto-resident, so the *raw* target arguments the compiler recorded are
available directly on the proto. This module surfaces them.

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

import io

from xprof.embedded.llo_analysis import llo_lite_pb2 as llo_module_pb2

# Fields of TpuTopologyArgsProto surfaced individually, best-effort. Accessed
# defensively via getattr because the exact field set can vary across the synced
# proto edition.
_TOPOLOGY_SCALAR_FIELDS = (
    "variant",
    "chip_config_name",
    "platform_type",
    "twist",
    "enhanced_barrier_enabled",
)
_TOPOLOGY_DIM_FIELDS = (
    "chips_per_host_bounds",
    "host_bounds",
)


def _fmt_dims(dims) -> str:
  """Formats a TpuDimensionsProto as an x/y/z[/w] tuple."""
  parts = [f"x={dims.x}", f"y={dims.y}", f"z={dims.z}"]
  # `w` has explicit presence; only show it when set.
  try:
    if dims.HasField("w"):
      parts.append(f"w={dims.w}")
  except (ValueError, AttributeError):
    pass
  return ", ".join(parts)


def extract_key_fields(
    module: llo_module_pb2.LloModuleProto,
) -> dict[str, object]:
  """Returns the proto-resident target fields as a flat dict (best-effort)."""
  out: dict[str, object] = {}
  if not module.HasField("target_arguments"):
    return out
  ta = module.target_arguments
  for field in ("replica_count", "reserved_hbm_usage_bytes"):
    try:
      out[field] = getattr(ta, field)
    except AttributeError:
      pass
  if ta.HasField("tpu_topology_args"):
    topo = ta.tpu_topology_args
    for field in _TOPOLOGY_SCALAR_FIELDS:
      try:
        out[field] = getattr(topo, field)
      except AttributeError:
        pass
    for field in _TOPOLOGY_DIM_FIELDS:
      try:
        if topo.HasField(field):
          out[field] = _fmt_dims(getattr(topo, field))
      except (ValueError, AttributeError):
        pass
    try:
      if topo.HasField("version"):
        # TpuVersionProto is a nested message; its text form is compact.
        out["version"] = str(topo.version).strip().replace("\n", " ")
    except (ValueError, AttributeError):
      pass
  return out


def render_target_info_markdown(
    module: llo_module_pb2.LloModuleProto,
) -> str:
  """Renders proto-resident target arguments + the documented catalog gap."""
  buf = io.StringIO()
  buf.write(f"# Target arguments: {module.hlo_instruction_name}\n\n")
  if not module.HasField("target_arguments"):
    buf.write("_No `target_arguments` present in this LLO module._\n")
    return buf.getvalue()

  fields = extract_key_fields(module)
  if fields:
    buf.write("## Key fields (proto-resident)\n\n")
    buf.write("| Field | Value |\n|---|---|\n")
    for key in sorted(fields):
      buf.write(f"| {key} | {fields[key]} |\n")
    buf.write("\n")

  buf.write("## Raw `target_arguments` (proto text)\n\n")
  buf.write("```\n")
  buf.write(str(module.target_arguments).rstrip() or "(empty)")
  buf.write("\n```\n\n")

  buf.write("## Not in the proto (requires an out-of-tree target catalog)\n\n")
  buf.write(
      "The following per-core hardware constants are NOT carried by"
      " `TargetArgumentsProto`; they are derived from the C++"
      " `TargetInfo::For(target_arguments)` catalog keyed by the TPU version"
      " and `chip_config_name`, so they are unavailable from the proto"
      " alone:\n\n- mxus / xlus per tensor core\n- VMEM / SMEM / CMEM sizes\n-"
      " tensor-core and HBM frequencies\n- HBM capacity per chip\n- tensor"
      " cores / sparse cores per chip\n"
  )
  return buf.getvalue()
