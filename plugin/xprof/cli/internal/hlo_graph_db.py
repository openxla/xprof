"""Scalable HLO Graph Database with CSR adjacency index and SQLite query engine.

Provides `HloGraphDb`, a dual-layer graph representation for XLA HLO modules:
1. In-memory Compressed Sparse Row (CSR) / integer-ID adjacency arrays for
   O(V_sub + E_sub) neighborhood traversal, shortest-path search, and call-tree
   expansion without materializing full-module HLO text strings.
2. Lazy in-memory SQLite relational tables (`computations`, `instructions`,
   `edges`, `call_edges`) for rich analytical queries required in AI model
   performance diagnosis: opcode aggregation, fusion-blocker detection, layout
   transition detection, critical-path hotspots, and arbitrary read-only SQL.
"""

import collections
import json
import math
import re
import sqlite3
import types
from typing import Any

from xprof.cli.internal import hlo_shape_utils

_OPERAND_RE = re.compile(r"(?:^|[\s,(])%?([a-zA-Z0-9._-]+)(?=[\s,)]|$)")

_ELEMENTWISE_FLOP_OPCODES: frozenset[str] = frozenset({
    "abs",
    "acos",
    "acosh",
    "add",
    "and",
    "asin",
    "asinh",
    "atan2",
    "atanh",
    "cbrt",
    "ceil",
    "clamp",
    "compare",
    "convert",
    "cos",
    "cosine",
    "cosh",
    "divide",
    "erf",
    "exp",
    "exponential",
    "exponential-minus-one",
    "expm1",
    "floor",
    "is-finite",
    "log",
    "log-plus-one",
    "log1p",
    "logistic",
    "maximum",
    "minimum",
    "multiply",
    "negate",
    "not",
    "or",
    "power",
    "remainder",
    "round-nearest-afz",
    "round-nearest-even",
    "rsqrt",
    "select",
    "shift-left",
    "shift-right-arithmetic",
    "shift-right-logical",
    "sign",
    "sin",
    "sine",
    "sinh",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "xor",
})

_ZERO_BYTES_OPCODES: frozenset[str] = frozenset({
    "after-all",
    "bitcast",
    "constant",
    "domain",
    "get-tuple-element",
    "optimization-barrier",
    "parameter",
    "partition-id",
    "replica-id",
    "tuple",
})


def _find_matching_paren(s: str, open_idx: int) -> int:
  """Returns the index of the closing parenthesis matching `s[open_idx]`."""
  depth = 0
  for i in range(open_idx, len(s)):
    ch = s[i]
    if ch == "(":
      depth += 1
    elif ch == ")":
      depth -= 1
      if depth == 0:
        return i
  return -1


# Primitive type integer to XLA string mapping (from xla_data.proto).
_PRIMITIVE_TYPE_NAMES: types.MappingProxyType[int, str] = (
    types.MappingProxyType({
        0: "invalid",
        1: "pred",
        2: "s8",
        3: "s16",
        4: "s32",
        5: "s64",
        6: "u8",
        7: "u16",
        8: "u32",
        9: "u64",
        10: "f16",
        11: "f32",
        12: "f64",
        13: "c64",
        14: "tuple",
        15: "opaque_type",
        16: "token",
        17: "bf16",
        18: "c128",
        19: "f8e5m2",
        20: "f8e4m3fn",
        21: "f8e4m3b11fnuz",
        22: "f8e5m2fnuz",
        23: "f8e4m3fnuz",
        24: "s4",
        25: "u4",
        26: "s2",
        27: "u2",
        28: "f8e3m4",
        29: "f8e4m3",
        30: "f8e8m0fnu",
        31: "f4e2m1fn",
    })
)

# Known XLA compiler fusion blocker opcodes (layout/type transitions).
_FUSION_BLOCKER_OPCODES: frozenset[str] = frozenset({
    "bitcast",
    "copy",
    "reshape",
    "convert",
})


def _shape_proto_metrics(shape_proto: Any) -> tuple[float, int, list[int]]:
  """Returns `(total_leaf_bytes, primary_elements, primary_dims)` for proto."""
  if shape_proto is None:
    return 0.0, 0, []
  elem_type = shape_proto.element_type
  if elem_type == 14:  # TUPLE
    total_bytes = 0.0
    best_elems = 0
    best_dims: list[int] = []
    for sub in shape_proto.tuple_shapes:
      sub_bytes, sub_elements, sub_dimensions = _shape_proto_metrics(sub)
      total_bytes += sub_bytes
      if (len(sub_dimensions), sub_elements) > (len(best_dims), best_elems):
        best_elems = sub_elements
        best_dims = sub_dimensions
    return total_bytes, best_elems, best_dims
  if elem_type in (0, 15, 16):  # INVALID, OPAQUE_TYPE, TOKEN
    return 0.0, 0, []
  dtype = _PRIMITIVE_TYPE_NAMES.get(elem_type, "f32")
  elem_bytes = hlo_shape_utils.DTYPE_BYTES.get(dtype, 4)
  raw_dims = shape_proto.dimensions
  if not raw_dims:
    return float(elem_bytes), 1, []
  dims = list(raw_dims)
  elems = math.prod(dims)
  return float(elem_bytes * elems), elems, dims


def _shape_str_metrics(shape_str: str) -> tuple[float, int, list[int]]:
  """Returns `(total_leaf_bytes, primary_elements, primary_dims)` for text."""
  if not shape_str or shape_str in ("unknown", "token[]", "()"):
    return 0.0, 0, []
  if not shape_str.startswith("("):
    bracket_idx = shape_str.find("[")
    if bracket_idx > 0:
      close_idx = shape_str.find("]", bracket_idx)
      dt = shape_str[:bracket_idx].lower()
      eb = hlo_shape_utils.DTYPE_BYTES.get(dt, 4)
      dims_part = (
          shape_str[bracket_idx + 1 : close_idx]
          if close_idx > bracket_idx
          else ""
      )
      if not dims_part:
        return float(eb), 1, []
      dims = [int(d) for d in dims_part.split(",") if d.strip().isdigit()]
      elems = math.prod(dims) if dims else 1
      return float(eb * elems), elems, dims
  matches = hlo_shape_utils.TENSOR_SHAPE_RE.findall(shape_str)
  if not matches:
    return 0.0, 0, []
  total_bytes = 0.0
  best_elems = 0
  best_dims: list[int] = []
  for dt, dims_str in matches:
    dims = [int(d.strip()) for d in dims_str.split(",") if d.strip().isdigit()]
    elems = math.prod(dims) if dims else 1
    eb = hlo_shape_utils.DTYPE_BYTES.get(dt.lower(), 4)
    total_bytes += float(eb * elems)
    if (len(dims), elems) > (len(best_dims), best_elems):
      best_elems = elems
      best_dims = dims
  return total_bytes, best_elems, best_dims


def format_shape_proto(shape_proto: Any) -> str:
  """Formats an XLA `ShapeProto` into a canonical HLO shape+layout string."""
  if shape_proto is None:
    return "unknown"
  elem_type = getattr(shape_proto, "element_type", 0)
  if elem_type == 14:  # TUPLE
    tuple_shapes = getattr(shape_proto, "tuple_shapes", ())
    inner = ", ".join(format_shape_proto(s) for s in tuple_shapes)
    return f"({inner})"
  if elem_type == 16:  # TOKEN
    return "token[]"
  dtype = _PRIMITIVE_TYPE_NAMES.get(elem_type, "f32")
  dims = getattr(shape_proto, "dimensions", ())
  dims_str = ",".join(str(d) for d in dims)
  layout_str = ""
  if hasattr(shape_proto, "HasField"):
    try:
      has_layout = shape_proto.HasField("layout")
    except ValueError:
      has_layout = bool(getattr(shape_proto, "layout", None))
  else:
    has_layout = bool(getattr(shape_proto, "layout", None))
  if has_layout:
    layout = shape_proto.layout
    m2m = getattr(layout, "minor_to_major", ())
    if m2m:
      m2m_str = ",".join(str(d) for d in m2m)
      tiles_part = ""
      tiles = getattr(layout, "tiles", ())
      if tiles:
        tile_strs = []
        for t in tiles:
          t_dims = getattr(t, "dimensions", ())
          if t_dims:
            tile_strs.append("(" + ",".join(str(d) for d in t_dims) + ")")
        if tile_strs:
          tiles_part = ":T" + "".join(tile_strs)
      mem_space = getattr(layout, "memory_space", 0)
      mem_part = f":S({mem_space})" if mem_space else ""
      layout_str = f"{{{m2m_str}{tiles_part}{mem_part}}}"
  return f"{dtype}[{dims_str}]{layout_str}"


def _extract_shape_dtype_layout(shape_str: str) -> tuple[str, str]:
  """Extracts `(dtype, layout)` from a formatted shape string in O(1)."""
  if not shape_str or shape_str == "unknown":
    return "", ""
  if shape_str.startswith("("):
    return "tuple", ""
  bracket_idx = shape_str.find("[")
  dtype = shape_str[:bracket_idx] if bracket_idx > 0 else shape_str
  brace_idx = shape_str.find("{")
  layout = shape_str[brace_idx:] if brace_idx >= 0 else ""
  return dtype, layout


def is_fusion_blocker(opcode: str, expression: str = "") -> bool:
  """Returns True if an instruction is a known fusion blocker."""
  op_clean = opcode.strip().lower()
  if op_clean in _FUSION_BLOCKER_OPCODES:
    return True
  if "AnalyzeLayout" in expression or "AllocateBuffer" in expression:
    return True
  return False


def get_fusion_blocker_reason(opcode: str, expression: str = "") -> str:
  """Returns a concise explanation of why an instruction blocks XLA fusion."""
  op_clean = opcode.strip().lower()
  if "AnalyzeLayout" in expression:
    return (
        "custom-call(@AnalyzeLayout) pins tensor layout and prevents"
        " producer-consumer loop fusion"
    )
  if "AllocateBuffer" in expression:
    return (
        "custom-call(@AllocateBuffer) materializes an explicit buffer"
        " allocation boundary"
    )
  if op_clean in ("bitcast", "copy", "reshape", "convert", "transpose"):
    return (
        f"{op_clean} changes layout/type or materializes a copy boundary"
        " between fusions"
    )
  if op_clean == "custom-call":
    target_match = re.search(r"custom_call_target=\"([^\"]+)\"", expression)
    if target_match:
      return (
          f"custom-call(@{target_match.group(1)}) executes an opaque kernel"
          " boundary"
      )
    return "custom-call executes an opaque non-fusable kernel boundary"
  if op_clean in (
      "all-gather",
      "all-gather-start",
      "all-gather-done",
      "all-reduce",
      "all-reduce-start",
      "all-reduce-done",
      "all-to-all",
      "collective-permute",
      "collective-permute-start",
      "collective-permute-done",
      "reduce-scatter",
  ):
    return (
        f"collective communication ({op_clean}) synchronizes across"
        " devices/ICI and breaks fusion"
    )
  if op_clean in ("optimization-barrier", "domain"):
    return f"{op_clean} explicitly barriers compiler scheduling and fusion"
  if op_clean in (
      "send",
      "send-done",
      "recv",
      "recv-done",
      "infeed",
      "outfeed",
  ):
    return f"host/channel I/O ({op_clean}) creates a side-effecting boundary"
  if op_clean == "sort":
    return (
        "sort requires multi-stage comparator reductions not fusable with"
        " arbitrary consumers"
    )
  if is_fusion_blocker(op_clean, expression):
    return f"{op_clean} prevents standard elementwise/loop fusion"
  return ""


class _TextInstruction:
  """Lightweight instruction record parsed from text HLO."""

  __slots__ = (
      "id",
      "name",
      "opcode",
      "shape_str",
      "dtype",
      "layout",
      "category",
      "expression",
      "operand_names",
      "called_comp_names",
      "metadata_op_name",
      "metadata_source_file",
      "metadata_source_line",
      "is_root",
  )

  def __init__(
      self,
      instr_id: int,
      name: str,
      opcode: str,
      shape_str: str,
      dtype: str,
      layout: str,
      expression: str,
      operand_names: list[str],
      called_comp_names: list[str],
      is_root: bool = False,
  ):
    self.id = instr_id
    self.name = name
    self.opcode = opcode
    self.shape_str = shape_str
    self.dtype = dtype
    self.layout = layout
    self.category = opcode
    self.expression = expression
    self.operand_names = operand_names
    self.called_comp_names = called_comp_names
    self.metadata_op_name = ""
    self.metadata_source_file = ""
    self.metadata_source_line = 0
    self.is_root = is_root


class HloGraphDb:
  """Unified in-memory CSR + SQLite relational HLO Graph Database."""

  def __init__(self, module_name: str = ""):
    self.module_name = module_name
    self.entry_computation_name: str = ""
    self.entry_computation_id: int = -1

    # CSR / Integer-ID Adjacency structures
    self.id_to_instr: dict[int, Any] = {}
    self.id_to_name: dict[int, str] = {}
    self.name_to_id: dict[str, int] = {}
    self.operands_by_id: dict[int, tuple[int, ...]] = {}
    self.users_by_id: dict[int, list[int]] = {}
    self.comp_name_by_id: dict[int, str] = {}
    self.comp_id_by_name: dict[str, int] = {}
    self.comp_instr_ids: dict[str, list[int]] = {}
    self.comp_root_id: dict[str, int] = {}
    self.called_comps_by_id: dict[int, tuple[str, ...]] = {}
    self.comp_callers: dict[str, list[int]] = {}

    # Lazy proto indexing structures
    self._is_proto_backed: bool = False
    self._hlo_module_proto: Any = None
    self._indexed_comp_names: set[str] = set()
    self._unindexed_comp_map: dict[str, Any] = {}
    self._unindexed_text_blocks: dict[str, list[str]] = {}
    self._next_text_instr_id: int = 1
    self._proto_comp_id_to_name: dict[int, str] = {}
    self._formatted_line_cache: dict[tuple[int, bool, bool], str] = {}
    self._shape_str_cache: dict[int, str] = {}
    self._sqlite_conn: sqlite3.Connection | None = None
    self._query_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

  @classmethod
  def from_hlo_module_proto(
      cls, hlo_proto: Any, module_name: str = ""
  ) -> "HloGraphDb":
    """Alias for `from_proto` accepting `HloModuleProto` or `HloProto`."""
    return cls.from_proto(hlo_proto, module_name=module_name)

  @classmethod
  def from_proto(cls, hlo_proto: Any, module_name: str = "") -> "HloGraphDb":
    """Constructs an `HloGraphDb` from an `HloProto` or `HloModuleProto`.

    Indexes the entry computation immediately so common entry-computation
    queries execute in O(V_entry) without touching non-entry computations,
    while lazily indexing non-entry computations on demand.
    """
    hlo_module = getattr(hlo_proto, "hlo_module", hlo_proto)
    mod_name = module_name or getattr(hlo_module, "name", "module")
    db = cls(module_name=mod_name)
    db._is_proto_backed = True
    db._hlo_module_proto = hlo_module
    db.entry_computation_name = getattr(
        hlo_module, "entry_computation_name", ""
    )
    db.entry_computation_id = getattr(hlo_module, "entry_computation_id", -1)

    computations = getattr(hlo_module, "computations", ())
    entry_comp = None
    for idx, comp in enumerate(computations):
      c_name = getattr(comp, "name", f"comp_{idx}")
      c_id = getattr(comp, "id", idx)
      db.comp_id_by_name[c_name] = c_id
      db._proto_comp_id_to_name[c_id] = c_name
      db._unindexed_comp_map[c_name] = comp
      if c_id == db.entry_computation_id or c_name == db.entry_computation_name:
        entry_comp = comp

    if entry_comp is None and computations:
      entry_comp = computations[-1]
      db.entry_computation_name = getattr(entry_comp, "name", "")
      db.entry_computation_id = getattr(entry_comp, "id", -1)

    if entry_comp is not None:
      db._index_proto_computation(entry_comp)

    return db

  def _index_proto_computation(self, comp: Any) -> None:
    """Indexes a single `HloComputationProto` into the CSR adjacency maps."""
    c_name = getattr(comp, "name", "")
    if not c_name or c_name in self._indexed_comp_names:
      return
    self._indexed_comp_names.add(c_name)
    self._unindexed_comp_map.pop(c_name, None)

    c_id = getattr(comp, "id", 0)
    self.comp_id_by_name[c_name] = c_id
    self._proto_comp_id_to_name[c_id] = c_name
    root_id = getattr(comp, "root_id", 0) or getattr(
        comp, "root_instruction_id", -1
    )
    self.comp_root_id[c_name] = root_id

    instr_ids: list[int] = []
    id_to_instr = self.id_to_instr
    id_to_name = self.id_to_name
    name_to_id = self.name_to_id
    operands_by_id = self.operands_by_id
    users_by_id = self.users_by_id
    comp_name_by_id = self.comp_name_by_id
    called_comps_by_id = self.called_comps_by_id
    comp_callers = self.comp_callers
    proto_comp_id_to_name = self._proto_comp_id_to_name

    for instr in getattr(comp, "instructions", ()):
      iid = instr.id
      raw_name = instr.name
      norm_name = raw_name[1:] if raw_name.startswith("%") else raw_name
      instr_ids.append(iid)
      id_to_instr[iid] = instr
      id_to_name[iid] = norm_name
      if norm_name not in name_to_id or c_name == self.entry_computation_name:
        name_to_id[norm_name] = iid
      comp_name_by_id[iid] = c_name

      op_ids = tuple(instr.operand_ids)
      operands_by_id[iid] = op_ids
      if iid not in users_by_id:
        users_by_id[iid] = []
      for op_id in op_ids:
        u_list = users_by_id.get(op_id)
        if u_list is None:
          users_by_id[op_id] = [iid]
        else:
          u_list.append(iid)

      called_ids = getattr(instr, "called_computation_ids", ())
      if called_ids:
        called_names = tuple(
            proto_comp_id_to_name[cid]
            for cid in called_ids
            if cid in proto_comp_id_to_name
        )
        called_comps_by_id[iid] = called_names
        for target_comp in called_names:
          callers = comp_callers.get(target_comp)
          if callers is None:
            comp_callers[target_comp] = [iid]
          else:
            callers.append(iid)
      else:
        called_comps_by_id[iid] = ()

    if root_id <= 0 and instr_ids:
      self.comp_root_id[c_name] = instr_ids[-1]
    self.comp_instr_ids[c_name] = instr_ids

  def _index_text_computation(
      self, c_name: str, lines: list[str], is_entry: bool = False
  ) -> None:
    """Indexes a single text HLO computation block into CSR maps."""
    if not c_name or c_name in self._indexed_comp_names:
      return
    self._indexed_comp_names.add(c_name)
    self._unindexed_text_blocks.pop(c_name, None)

    comp_instrs: list[_TextInstruction] = []
    instr_ids: list[int] = []
    root_id = -1
    next_instr_id = self._next_text_instr_id

    for line in lines:
      is_root = line.startswith("ROOT ")
      expr_body = line[5:].strip() if is_root else line
      if "=" not in expr_body:
        continue
      lhs, rhs = expr_body.split("=", 1)
      lhs = lhs.strip()
      norm_name = lhs[1:] if lhs.startswith("%") else lhs
      rhs = rhs.strip()

      opcode = "unknown"
      shape_str = "unknown"
      if rhs.startswith("("):
        tuple_end = _find_matching_paren(rhs, 0)
        paren_idx = rhs.find("(", tuple_end + 1) if tuple_end != -1 else -1
      else:
        paren_idx = rhs.find("(")
      if paren_idx != -1:
        prefix = rhs[:paren_idx].strip()
        tokens = prefix.split()
        if len(tokens) >= 2:
          opcode = tokens[-1]
          shape_str = " ".join(tokens[:-1])
        elif len(tokens) == 1:
          opcode = tokens[0]

      dtype, layout = _extract_shape_dtype_layout(shape_str)
      args_str = ""
      if paren_idx != -1:
        close_idx = _find_matching_paren(rhs, paren_idx)
        if close_idx != -1:
          args_str = rhs[paren_idx + 1 : close_idx]
      operand_names = _OPERAND_RE.findall(args_str)

      called_comp_names: list[str] = []
      if (
          "calls=" in rhs
          or "to_apply=" in rhs
          or "body=" in rhs
          or "condition=" in rhs
          or "computation=" in rhs
      ):
        for attr in (
            "to_apply",
            "calls",
            "condition",
            "body",
            "true_computation",
            "false_computation",
        ):
          if attr in rhs:
            for m in re.finditer(rf"{attr}=%?([^\s,}})]+)", rhs):
              called_comp_names.append(m.group(1).lstrip("%"))

      iid = next_instr_id
      next_instr_id += 1
      rec = _TextInstruction(
          instr_id=iid,
          name=norm_name,
          opcode=opcode,
          shape_str=shape_str,
          dtype=dtype,
          layout=layout,
          expression=line,
          operand_names=operand_names,
          called_comp_names=called_comp_names,
          is_root=is_root,
      )
      comp_instrs.append(rec)
      instr_ids.append(iid)
      self.id_to_instr[iid] = rec
      self.id_to_name[iid] = norm_name
      if norm_name not in self.name_to_id or is_entry:
        self.name_to_id[norm_name] = iid
      self.comp_name_by_id[iid] = c_name
      self.users_by_id[iid] = []
      if is_root:
        root_id = iid

    self._next_text_instr_id = next_instr_id
    if root_id == -1 and instr_ids:
      root_id = instr_ids[-1]
    self.comp_root_id[c_name] = root_id
    self.comp_instr_ids[c_name] = instr_ids

    local_name_to_id = {rec.name: rec.id for rec in comp_instrs}
    for rec in comp_instrs:
      op_ids: list[int] = []
      for op_name in rec.operand_names:
        op_id = local_name_to_id.get(op_name) or self.name_to_id.get(op_name)
        if op_id is not None:
          op_ids.append(op_id)
          self.users_by_id.setdefault(op_id, []).append(rec.id)
      self.operands_by_id[rec.id] = tuple(op_ids)
      called_tuple = tuple(
          cc for cc in rec.called_comp_names if cc in self.comp_id_by_name
      )
      self.called_comps_by_id[rec.id] = called_tuple
      for cc in called_tuple:
        self.comp_callers.setdefault(cc, []).append(rec.id)

  def _ensure_all_indexed(self) -> None:
    """Indexes all remaining unindexed computations on demand."""
    if self._unindexed_comp_map:
      for comp in list(self._unindexed_comp_map.values()):
        self._index_proto_computation(comp)
    if self._unindexed_text_blocks:
      for c_name, lines in list(self._unindexed_text_blocks.items()):
        self._index_text_computation(
            c_name, lines, is_entry=(c_name == self.entry_computation_name)
        )

  @classmethod
  def from_hlo_text(cls, hlo_text: str, module_name: str = "") -> "HloGraphDb":
    """Alias for `from_text`."""
    return cls.from_text(hlo_text, module_name=module_name)

  @classmethod
  def from_text(cls, hlo_text: str, module_name: str = "") -> "HloGraphDb":
    """Constructs an `HloGraphDb` from an HLO text dump lazily."""
    header_match = re.search(r"HloModule\s+([^\s,(:]+)", hlo_text)
    mod_name = module_name or (
        header_match.group(1) if header_match else "module"
    )
    db = cls(module_name=mod_name)
    db._is_proto_backed = False

    comp_idx = 0
    current_comp = ""
    current_lines: list[str] = []
    comp_blocks: list[tuple[str, bool, list[str]]] = []

    for line in hlo_text.splitlines():
      stripped = line.strip()
      if not stripped:
        continue
      if (
          stripped.startswith("ENTRY ")
          or stripped.startswith("%")
          or (stripped[0].isalpha() and "(" in stripped)
      ) and (stripped.endswith("{") or stripped.endswith("(")):
        is_entry = stripped.startswith("ENTRY ")
        header_part = stripped[6:].strip() if is_entry else stripped
        name_part = header_part.split("(", 1)[0].split("{", 1)[0].strip()
        c_name = name_part[1:] if name_part.startswith("%") else name_part
        if c_name and not c_name.startswith("HloModule"):
          current_comp = c_name
          current_lines = []
          if is_entry or not db.entry_computation_name:
            db.entry_computation_name = c_name
            db.entry_computation_id = comp_idx
          db.comp_id_by_name[c_name] = comp_idx
          comp_idx += 1
          continue
      if current_comp:
        if stripped in ("}", ")"):
          comp_blocks.append((
              current_comp,
              current_comp == db.entry_computation_name,
              current_lines,
          ))
          current_comp = ""
          current_lines = []
        else:
          current_lines.append(stripped)

    if not comp_blocks and hlo_text.strip():
      c_name = "main"
      db.entry_computation_name = c_name
      db.entry_computation_id = 0
      db.comp_id_by_name[c_name] = 0
      comp_blocks.append((
          c_name,
          True,
          [
              l.strip()
              for l in hlo_text.splitlines()
              if l.strip()
              and not l.strip().startswith("HloModule")
              and l.strip() not in ("}", ")")
          ],
      ))

    for c_name, _, lines in comp_blocks:
      if c_name == db.entry_computation_name:
        db._index_text_computation(c_name, lines, is_entry=True)
      else:
        db._unindexed_text_blocks[c_name] = lines

    return db

  def resolve_node_id(self, identifier: str | int) -> int | None:
    """Resolves an instruction name (with or without `%`) or ID."""
    if isinstance(identifier, int):
      if identifier in self.id_to_instr:
        return identifier
      self._ensure_all_indexed()
      return identifier if identifier in self.id_to_instr else None

    clean = identifier.strip()
    if clean.startswith("%"):
      clean = clean[1:]

    candidates = [clean]
    if "_" in clean:
      candidates.append(clean.replace("_", "."))
      parts = clean.rsplit("_", 1)
      if len(parts) == 2 and parts[1].isdigit():
        candidates.append(f"{parts[0]}.{parts[1]}")
    if "." in clean:
      candidates.append(clean.replace(".", "_"))

    for cand in candidates:
      if cand in self.name_to_id:
        return self.name_to_id[cand]

    while self._unindexed_comp_map:
      _, comp = next(iter(self._unindexed_comp_map.items()))
      self._index_proto_computation(comp)
      for cand in candidates:
        if cand in self.name_to_id:
          return self.name_to_id[cand]

    while self._unindexed_text_blocks:
      c_name, lines = next(iter(self._unindexed_text_blocks.items()))
      self._index_text_computation(c_name, lines)
      for cand in candidates:
        if cand in self.name_to_id:
          return self.name_to_id[cand]

    if clean.isdigit():
      int_id = int(clean)
      if int_id in self.id_to_instr:
        return int_id
    return None

  def _get_shape_str(self, iid: int, instr: Any) -> str:
    """Returns cached formatted shape string for `iid`."""
    cached = self._shape_str_cache.get(iid)
    if cached is not None:
      return cached
    if isinstance(instr, _TextInstruction):
      res = instr.shape_str
    else:
      res = format_shape_proto(getattr(instr, "shape", None))
    self._shape_str_cache[iid] = res
    return res

  def _get_opcode(self, iid: int) -> str:
    instr = self.id_to_instr.get(iid)
    if instr is None:
      return "unknown"
    return getattr(instr, "opcode", "unknown")

  def format_instruction(
      self,
      iid: int,
      show_metadata: bool = False,
      annotate_fusion_blockers: bool = False,
      name_to_line_override: dict[str, str] | None = None,
  ) -> str:
    """Formats a single instruction into canonical HLO text in O(1)."""
    norm_name = self.id_to_name.get(iid, str(iid))
    opcode = self._get_opcode(iid)

    if name_to_line_override:
      override_line = name_to_line_override.get(
          norm_name
      ) or name_to_line_override.get(f"%{norm_name}")
      if override_line:
        base_line = override_line.strip()
        if annotate_fusion_blockers and is_fusion_blocker(opcode, base_line):
          base_line = f"{base_line} [FUSION_BLOCKER: {opcode}]"
        return base_line

    cache_key = (iid, show_metadata, annotate_fusion_blockers)
    cached = self._formatted_line_cache.get(cache_key)
    if cached is not None:
      return cached

    instr = self.id_to_instr.get(iid)
    if instr is None:
      return ""

    comp_name = self.comp_name_by_id.get(iid, "")
    is_root = self.comp_root_id.get(comp_name) == iid
    root_prefix = "ROOT " if is_root else ""

    if isinstance(instr, _TextInstruction):
      base_line = instr.expression
      if annotate_fusion_blockers and is_fusion_blocker(
          instr.opcode, base_line
      ):
        base_line = f"{base_line} [FUSION_BLOCKER: {instr.opcode}]"
      self._formatted_line_cache[cache_key] = base_line
      return base_line

    shape_str = self._get_shape_str(iid, instr)
    op_ids = self.operands_by_id.get(iid, ())
    op_names = ", ".join(
        f"%{self.id_to_name.get(oid, str(oid))}" for oid in op_ids
    )

    attrs: list[str] = []
    if opcode == "parameter":
      attrs.append(str(getattr(instr, "parameter_number", 0)))
    elif op_names:
      attrs.append(op_names)

    called_comps = self.called_comps_by_id.get(iid, ())
    if called_comps:
      if opcode == "fusion":
        fusion_kind = getattr(instr, "fusion_kind", "")
        if fusion_kind:
          attrs.append(f"kind={fusion_kind}")
        attrs.append(f"calls=%{called_comps[0]}")
      else:
        attrs.append(f"calls=%{called_comps[0]}")

    custom_target = getattr(instr, "custom_call_target", "")
    if custom_target:
      attrs.append(f'custom_call_target="{custom_target}"')

    backend_cfg = getattr(instr, "backend_config", b"")
    if backend_cfg:
      if isinstance(backend_cfg, bytes):
        cfg_str = backend_cfg.decode("utf-8", errors="ignore")
      else:
        cfg_str = str(backend_cfg)
      if cfg_str:
        attrs.append(f"backend_config={cfg_str}")

    if show_metadata and hasattr(instr, "metadata"):
      meta = instr.metadata
      op_name_meta = getattr(meta, "op_name", "")
      src_file = getattr(meta, "source_file", "")
      src_line = getattr(meta, "source_line", 0)
      if op_name_meta:
        attrs.append(f'metadata={{op_name="{op_name_meta}"}}')
      if src_file:
        attrs.append(f'source_file="{src_file}:{src_line}"')

    line = (
        f"{root_prefix}%{norm_name} = {shape_str} {opcode}({', '.join(attrs)})"
    )
    if annotate_fusion_blockers and is_fusion_blocker(opcode, line):
      line = f"{line} [FUSION_BLOCKER: {opcode}]"

    self._formatted_line_cache[cache_key] = line
    return line

  def get_neighborhood(
      self,
      node_name: str,
      radius: int = 2,
      fmt: str = "text",
      *,
      hops: int | None = None,
      print_metadata: bool = False,
      show_metadata: bool = False,
      direction: str = "both",
      follow_calls: bool = False,
      include_called_computations: bool = False,
      detect_fusion_blockers: bool = False,
      annotate_fusion_blockers: bool = False,
      opcode_filter: str | None = None,
      name_to_line_override: dict[str, str] | None = None,
      oss_style_suggestions: bool = False,
      max_nodes: int = 200,
  ) -> str:
    """Extracts a k-hop neighborhood around `node_name` in O(V_sub + E_sub)."""
    del oss_style_suggestions
    effective_radius = hops if hops is not None else radius
    effective_metadata = print_metadata or show_metadata
    effective_follow_calls = follow_calls or include_called_computations
    effective_blockers = detect_fusion_blockers or annotate_fusion_blockers

    center_id = self.resolve_node_id(node_name)
    if center_id is None:
      clean_name = node_name[1:] if node_name.startswith("%") else node_name
      return f"Instruction '{clean_name}' not found in HLO module."

    resolved_name = self.id_to_name[center_id]
    dir_norm = (direction or "both").strip().lower()
    want_parents = dir_norm in (
        "all",
        "both",
        "parents",
        "up",
        "upstream",
        "operands",
    )
    want_children = dir_norm in (
        "all",
        "both",
        "children",
        "down",
        "downstream",
        "users",
    )

    visited: dict[int, int] = {center_id: 0}
    queue: collections.deque[tuple[int, int]] = collections.deque(
        [(center_id, 0)]
    )
    called_comps_to_expand: set[str] = set()

    while queue and len(visited) < max_nodes:
      curr_id, dist = queue.popleft()
      if effective_follow_calls:
        for cc in self.called_comps_by_id.get(curr_id, ()):
          called_comps_to_expand.add(cc)

      if dist >= effective_radius:
        continue

      neighbors: list[int] = []
      if want_parents:
        neighbors.extend(self.operands_by_id.get(curr_id, ()))
      if want_children:
        neighbors.extend(self.users_by_id.get(curr_id, ()))

      for nb_id in neighbors:
        if nb_id not in visited:
          visited[nb_id] = dist + 1
          queue.append((nb_id, dist + 1))
          if len(visited) >= max_nodes:
            break

    if effective_follow_calls and called_comps_to_expand:
      for cc_name in sorted(called_comps_to_expand):
        if cc_name in self._unindexed_comp_map:
          self._index_proto_computation(self._unindexed_comp_map[cc_name])
        elif cc_name in self._unindexed_text_blocks:
          self._index_text_computation(
              cc_name, self._unindexed_text_blocks[cc_name]
          )
        for cc_instr_id in self.comp_instr_ids.get(cc_name, ()):
          if len(visited) >= max_nodes:
            break
          if cc_instr_id not in visited:
            visited[cc_instr_id] = effective_radius + 1

    if isinstance(self.id_to_instr.get(center_id), _TextInstruction):
      ordered_items = sorted(
          visited.items(),
          key=lambda kv: (kv[1], self.id_to_name.get(kv[0], "")),
      )
    else:
      ordered_items = sorted(visited.items(), key=lambda kv: (kv[1], kv[0]))
    if opcode_filter:
      filt_low = opcode_filter.lower()
      ordered_items = [
          (iid, dist)
          for iid, dist in ordered_items
          if iid == center_id or filt_low in self._get_opcode(iid).lower()
      ]

    if fmt == "markdown":
      lines = [
          f"### Neighborhood of '{resolved_name}' (radius={effective_radius}):"
      ]
      for iid, dist in ordered_items:
        comp_name = self.comp_name_by_id.get(iid, self.entry_computation_name)
        line_str = self.format_instruction(
            iid,
            show_metadata=effective_metadata,
            annotate_fusion_blockers=effective_blockers,
            name_to_line_override=name_to_line_override,
        )
        lines.append(f"- [dist={dist}] [{comp_name}] {line_str}")
      return "\n".join(lines)

    lines = [f"Neighborhood of '{resolved_name}' (radius={effective_radius}):"]
    for iid, dist in ordered_items:
      comp_name = self.comp_name_by_id.get(iid, self.entry_computation_name)
      line_str = self.format_instruction(
          iid,
          show_metadata=effective_metadata,
          annotate_fusion_blockers=effective_blockers,
          name_to_line_override=name_to_line_override,
      )
      lines.append(f"  [dist={dist}] [{comp_name}] {line_str}")
    return "\n".join(lines)

  def find_shortest_path(
      self, source: str, target: str, max_depth: int = 50
  ) -> dict[str, Any]:
    """Finds a directed or undirected shortest path between ops."""
    src_id = self.resolve_node_id(source)
    dst_id = self.resolve_node_id(target)
    if src_id is None or dst_id is None:
      return {
          "status": "SUCCESS",
          "found": False,
          "error": (
              f"Could not resolve source={source!r} (id={src_id}) or"
              f" target={target!r} (id={dst_id})."
          ),
      }
    if src_id == dst_id:
      return {
          "status": "SUCCESS",
          "found": True,
          "distance": 0,
          "direction": "self",
          "path": [{
              "id": src_id,
              "name": self.id_to_name[src_id],
              "opcode": self._get_opcode(src_id),
              "computation": self.comp_name_by_id.get(src_id, ""),
              "expression": self.format_instruction(src_id),
          }],
      }

    def _bfs(directed_mode: str) -> list[tuple[int, str]] | None:
      queue: collections.deque[tuple[int, int]] = collections.deque(
          [(src_id, 0)]
      )
      parent_map: dict[int, tuple[int, str]] = {src_id: (-1, "start")}
      while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
          continue
        next_edges: list[tuple[int, str]] = []
        if directed_mode in ("downstream", "undirected"):
          for user_id in self.users_by_id.get(curr, ()):
            next_edges.append((user_id, "-> (used_by)"))
          for cc in self.called_comps_by_id.get(curr, ()):
            if cc in self._unindexed_comp_map:
              self._index_proto_computation(self._unindexed_comp_map[cc])
            elif cc in self._unindexed_text_blocks:
              self._index_text_computation(cc, self._unindexed_text_blocks[cc])
            r_id = self.comp_root_id.get(cc, -1)
            if r_id != -1:
              next_edges.append((r_id, f"-> (calls %{cc})"))
        if directed_mode in ("upstream", "undirected"):
          for op in self.operands_by_id.get(curr, ()):
            next_edges.append((op, "<- (operand)"))
          c_name = self.comp_name_by_id.get(curr, "")
          for caller_id in self.comp_callers.get(c_name, ()):
            next_edges.append((
                caller_id,
                f"<- (called_from %{self.id_to_name.get(caller_id, '')})",
            ))

        for nb_id, rel in next_edges:
          if nb_id not in parent_map:
            parent_map[nb_id] = (curr, rel)
            if nb_id == dst_id:
              path_rev: list[tuple[int, str]] = []
              node = dst_id
              while node != -1:
                p_node, edge_rel = parent_map[node]
                path_rev.append((node, edge_rel))
                node = p_node
              path_rev.reverse()
              return path_rev
            queue.append((nb_id, depth + 1))
      return None

    for mode_label in ("downstream", "upstream", "undirected"):
      res_path = _bfs(mode_label)
      if res_path is not None:
        steps = []
        for iid, rel in res_path:
          steps.append({
              "id": iid,
              "name": self.id_to_name[iid],
              "opcode": self._get_opcode(iid),
              "computation": self.comp_name_by_id.get(iid, ""),
              "relation": rel,
              "expression": self.format_instruction(
                  iid, annotate_fusion_blockers=True
              ),
          })
        return {
            "status": "SUCCESS",
            "found": True,
            "distance": len(steps) - 1,
            "direction": mode_label,
            "path": steps,
        }

    return {
        "status": "SUCCESS",
        "found": False,
        "error": (
            f"No path found between %{self.id_to_name[src_id]} and"
            f" %{self.id_to_name[dst_id]} within {max_depth} hops."
        ),
    }

  def _compute_module_costs(
      self,
  ) -> tuple[dict[int, float], dict[int, float]]:
    """Computes per-instruction `(flops_by_id, bytes_by_id)`."""
    out_bytes: dict[int, float] = {}
    out_elems: dict[int, int] = {}
    out_dims: dict[int, list[int]] = {}

    for iid, instr in self.id_to_instr.items():
      if isinstance(instr, _TextInstruction):
        b_val, el_val, d_val = _shape_str_metrics(instr.shape_str)
      else:
        b_val, el_val, d_val = _shape_proto_metrics(instr.shape)
      out_bytes[iid] = b_val
      out_elems[iid] = el_val
      out_dims[iid] = d_val

    flops_by_id: dict[int, float] = {}
    bytes_by_id: dict[int, float] = {}
    operands_by_id = self.operands_by_id
    id_to_name = self.id_to_name
    id_to_instr = self.id_to_instr

    for iid, instr in id_to_instr.items():
      op_ids = operands_by_id.get(iid, ())
      opcode = instr.opcode
      op_clean = opcode.lower()

      if op_clean in _ZERO_BYTES_OPCODES:
        b_acc = 0.0
      elif op_clean in ("slice", "dynamic-slice"):
        b_acc = 2.0 * out_bytes[iid]
      elif op_clean == "dynamic-update-slice":
        upd_b = (
            out_bytes.get(op_ids[1], out_bytes[iid])
            if len(op_ids) > 1
            else out_bytes[iid]
        )
        b_acc = 2.0 * upd_b
      elif op_clean == "infeed":
        b_acc = out_bytes[iid]
      elif op_clean == "outfeed":
        b_acc = sum(out_bytes.get(oid, 0.0) for oid in op_ids)
      else:
        b_acc = out_bytes[iid] + sum(out_bytes.get(oid, 0.0) for oid in op_ids)

      f_val = 0.0
      if op_clean in _ELEMENTWISE_FLOP_OPCODES:
        f_val = (
            float(3 * out_elems[iid])
            if op_clean == "clamp"
            else float(out_elems[iid])
        )
      elif op_clean in ("dot", "ragged-dot", "scaled-dot"):
        dnums = getattr(instr, "dot_dimension_numbers", None)
        lhs_contracting = (
            getattr(dnums, "lhs_contracting_dimensions", ()) if dnums else ()
        )
        if lhs_contracting and op_ids:
          lhs_d = out_dims.get(op_ids[0], [])
          red_w = math.prod(
              lhs_d[d] for d in lhs_contracting if 0 <= d < len(lhs_d)
          )
          f_val = float(2 * out_elems[iid] * red_w)
        elif len(op_ids) >= 2:
          k_dim = hlo_shape_utils.try_match_contraction(
              out_dims[iid],
              out_dims.get(op_ids[0], []),
              out_dims.get(op_ids[1], []),
          )
          if k_dim is not None:
            f_val = float(2 * out_elems[iid] * k_dim)
      elif op_clean == "convolution":
        if len(op_ids) >= 2:
          rhs_el = out_elems.get(op_ids[1], 1)
          out_c = out_dims[iid][-1] if out_dims[iid] else 1
          f_val = float(2 * out_elems[iid] * max(1, rhs_el // max(1, out_c)))
      elif op_clean in ("reduce", "reduce-window"):
        if op_ids:
          in_el = out_elems.get(op_ids[0], out_elems[iid])
          f_val = float(max(0, in_el - out_elems[iid]))
      elif "custom-call" in op_clean or "custom_call" in op_clean:
        norm_name = id_to_name[iid]
        shape_str = self._get_shape_str(iid, instr)
        op_shape_strs = [
            f"{self._get_shape_str(oid, id_to_instr[oid])}"
            f" %{id_to_name.get(oid, str(oid))}"
            for oid in op_ids
            if oid in id_to_instr
        ]
        expr_with_shapes = (
            f"%{norm_name} = {shape_str} {opcode}({', '.join(op_shape_strs)})"
        )
        c_flops, c_bytes, prov = (
            hlo_shape_utils.derive_custom_call_flops_and_bytes(
                expr_with_shapes, opcode, norm_name
            )
        )
        if prov == "derived_from_shapes" and c_flops is not None:
          f_val = float(c_flops)
          if c_bytes is not None:
            b_acc = float(c_bytes)

      flops_by_id[iid] = f_val
      bytes_by_id[iid] = b_acc

    for _ in range(2):
      comp_flops: dict[str, float] = {}
      for c_name, iids in self.comp_instr_ids.items():
        comp_flops[c_name] = sum(flops_by_id.get(i, 0.0) for i in iids)
      for iid, called_comps in self.called_comps_by_id.items():
        if called_comps and flops_by_id.get(iid, 0.0) <= 0.0:
          flops_by_id[iid] = sum(comp_flops.get(cc, 0.0) for cc in called_comps)

    return flops_by_id, bytes_by_id

  def _ensure_sqlite_db(self) -> sqlite3.Connection:
    """Lazily materializes the in-memory SQLite relational database."""
    if self._sqlite_conn is not None:
      return self._sqlite_conn

    # Unconditionally index all computations for SQLite queries.
    self._ensure_all_indexed()

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
      CREATE TABLE computations (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        is_entry INTEGER,
        instruction_count INTEGER,
        root_id INTEGER
      );

      CREATE TABLE instructions (
        id INTEGER PRIMARY KEY,
        name TEXT,
        computation_name TEXT,
        opcode TEXT,
        category TEXT,
        shape TEXT,
        dtype TEXT,
        layout TEXT,
        is_root INTEGER,
        is_fusion_blocker INTEGER,
        operand_count INTEGER,
        user_count INTEGER,
        operands_json TEXT,
        called_computations_json TEXT,
        metadata_op_name TEXT,
        metadata_source_file TEXT,
        metadata_source_line INTEGER,
        flops REAL,
        bytes_accessed REAL,
        optimal_seconds REAL,
        expression TEXT
      );

      CREATE TABLE edges (
        src_id INTEGER,
        dst_id INTEGER,
        operand_index INTEGER
      );

      CREATE TABLE call_edges (
        caller_instr_id INTEGER,
        caller_comp_name TEXT,
        callee_comp_name TEXT
      );

      CREATE INDEX idx_instr_name ON instructions(name);
      CREATE INDEX idx_instr_opcode ON instructions(opcode);
      CREATE INDEX idx_instr_comp ON instructions(computation_name);
      CREATE INDEX idx_instr_blocker ON instructions(is_fusion_blocker);
      CREATE INDEX idx_edges_src ON edges(src_id);
      CREATE INDEX idx_edges_dst ON edges(dst_id);
    """)

    comp_rows = []
    for c_name, c_id in self.comp_id_by_name.items():
      iids = self.comp_instr_ids.get(c_name, [])
      comp_rows.append((
          c_id,
          c_name,
          1 if c_name == self.entry_computation_name else 0,
          len(iids),
          self.comp_root_id.get(c_name, -1),
      ))
    conn.executemany(
        "INSERT INTO computations VALUES (?, ?, ?, ?, ?)", comp_rows
    )

    flops_by_id, bytes_by_id = self._compute_module_costs()

    instr_rows = []
    edge_rows = []
    call_edge_rows = []

    for iid, instr in self.id_to_instr.items():
      norm_name = self.id_to_name[iid]
      c_name = self.comp_name_by_id.get(iid, "")
      is_root = 1 if self.comp_root_id.get(c_name) == iid else 0
      op_ids = self.operands_by_id.get(iid, ())
      u_ids = self.users_by_id.get(iid, ())
      called_comps = self.called_comps_by_id.get(iid, ())

      if isinstance(instr, _TextInstruction):
        opcode = instr.opcode
        category = instr.category
        shape_str = instr.shape_str
        dtype_str = instr.dtype
        layout_str = instr.layout
        expr = instr.expression
        meta_op = instr.metadata_op_name
        meta_file = instr.metadata_source_file
        meta_line = instr.metadata_source_line
        opt_sec = 0.0
      else:
        opcode = getattr(instr, "opcode", "unknown")
        category = getattr(instr, "category", "") or opcode
        shape_str = self._get_shape_str(iid, instr)
        dtype_str, layout_str = _extract_shape_dtype_layout(shape_str)
        expr = self.format_instruction(iid)
        meta = getattr(instr, "metadata", None)
        meta_op = getattr(meta, "op_name", "") if meta else ""
        meta_file = getattr(meta, "source_file", "") if meta else ""
        meta_line = getattr(meta, "source_line", 0) if meta else 0
        opt_sec = float(getattr(instr, "optimal_seconds", 0.0) or 0.0)

      flops = flops_by_id.get(iid, 0.0)
      bytes_acc = bytes_by_id.get(iid, 0.0)

      blocker_flag = 1 if is_fusion_blocker(opcode, expr) else 0
      op_names_json = json.dumps(
          [f"%{self.id_to_name.get(oid, str(oid))}" for oid in op_ids]
      )
      called_json = json.dumps(list(called_comps))

      instr_rows.append((
          iid,
          norm_name,
          c_name,
          opcode,
          category,
          shape_str,
          dtype_str,
          layout_str,
          is_root,
          blocker_flag,
          len(op_ids),
          len(u_ids),
          op_names_json,
          called_json,
          meta_op,
          meta_file,
          meta_line,
          flops,
          bytes_acc,
          opt_sec,
          expr,
      ))

      for op_idx, src_id in enumerate(op_ids):
        edge_rows.append((src_id, iid, op_idx))

      for callee in called_comps:
        call_edge_rows.append((iid, c_name, callee))

    conn.executemany(
        "INSERT INTO instructions VALUES ("
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
        ")",
        instr_rows,
    )
    conn.executemany("INSERT INTO edges VALUES (?, ?, ?)", edge_rows)
    conn.executemany("INSERT INTO call_edges VALUES (?, ?, ?)", call_edge_rows)
    conn.commit()
    conn.execute("PRAGMA query_only = ON")

    self._sqlite_conn = conn
    return conn

  def query(
      self,
      mode: str = "summary",
      *,
      opcode: str | None = None,
      category: str | None = None,
      comp_name: str | None = None,
      computation: str | None = None,
      name_pattern: str | None = None,
      filter_str: str | None = None,
      src_op: str | None = None,
      dst_op: str | None = None,
      source: str | None = None,
      target: str | None = None,
      sql: str | None = None,
      limit: int = 50,
      sort_by: str = "count",
  ) -> dict[str, Any]:
    """Executes an analytical HLO graph query and returns a structured dict."""
    eff_comp = comp_name or computation
    eff_filter = name_pattern or filter_str
    eff_src = src_op or source
    eff_dst = dst_op or target

    cache_key = (
        mode,
        eff_filter,
        opcode,
        eff_comp,
        category,
        eff_src,
        eff_dst,
        sql,
        limit,
        sort_by,
    )
    cached_res = self._query_cache.get(cache_key)
    if cached_res is not None:
      return cached_res

    mode_clean = (mode or "summary").strip().lower()

    if mode_clean == "shortest_path":
      if not eff_src or not eff_dst:
        raise ValueError(
            "mode='shortest_path' requires both `src_op`/`source` and"
            " `dst_op`/`target`."
        )
      res = self.find_shortest_path(eff_src, eff_dst, max_depth=max(limit, 50))
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "shortest_path",
          **res,
      }
      self._query_cache[cache_key] = out
      return out

    conn = self._ensure_sqlite_db()
    cursor = conn.cursor()

    if mode_clean == "sql":
      if not sql:
        raise ValueError("mode='sql' requires a `sql` SELECT query string.")
      sql_stripped = sql.strip()
      first_word = (
          sql_stripped.split(None, 1)[0].upper() if sql_stripped else ""
      )
      if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA") or (
          first_word == "PRAGMA" and "=" in sql_stripped
      ):
        raise ValueError(
            "Only read-only SELECT/WITH queries are allowed in mode='sql'."
        )
      try:
        cursor.execute(sql_stripped)
        rows = [dict(r) for r in cursor.fetchmany(limit)]
      except sqlite3.OperationalError as e:
        raise ValueError(
            f"Only read-only SELECT/WITH queries are allowed in mode='sql': {e}"
        ) from e
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "sql",
          "row_count": len(rows),
          "rows": rows,
      }
      self._query_cache[cache_key] = out
      return out

    if mode_clean == "summary":
      cursor.execute("SELECT COUNT(*) AS cnt FROM computations")
      comp_count = cursor.fetchone()["cnt"]
      cursor.execute("SELECT COUNT(*) AS cnt FROM instructions")
      instr_count = cursor.fetchone()["cnt"]
      cursor.execute("SELECT COUNT(*) AS cnt FROM edges")
      edge_count = cursor.fetchone()["cnt"]
      cursor.execute(
          "SELECT COUNT(*) AS cnt FROM instructions WHERE opcode = 'fusion'"
      )
      fusion_count = cursor.fetchone()["cnt"]
      cursor.execute(
          "SELECT COUNT(*) AS cnt FROM instructions WHERE opcode ="
          " 'custom-call'"
      )
      custom_call_count = cursor.fetchone()["cnt"]
      cursor.execute(
          "SELECT COUNT(*) AS cnt FROM instructions WHERE is_fusion_blocker = 1"
      )
      blocker_count = cursor.fetchone()["cnt"]
      cursor.execute(
          """
        SELECT opcode, COUNT(*) AS count,
               SUM(bytes_accessed) AS total_bytes, SUM(flops) AS total_flops
        FROM instructions
        GROUP BY opcode
        ORDER BY count DESC
        LIMIT ?
      """,
          (min(limit, 15),),
      )
      top_opcodes = [dict(r) for r in cursor.fetchall()]
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "summary",
          "entry_computation": self.entry_computation_name,
          "computation_count": comp_count,
          "instruction_count": instr_count,
          "edge_count": edge_count,
          "fusion_count": fusion_count,
          "custom_call_count": custom_call_count,
          "fusion_blocker_count": blocker_count,
          "top_opcodes": top_opcodes,
      }
      self._query_cache[cache_key] = out
      return out

    if mode_clean == "opcode_stats":
      where_clause = ""
      params: list[Any] = []
      if eff_comp:
        where_clause = "WHERE computation_name = ?"
        params.append(eff_comp.lstrip("%"))
      order_col = "count DESC"
      if sort_by == "bytes":
        order_col = "total_bytes DESC, count DESC"
      elif sort_by == "flops":
        order_col = "total_flops DESC, count DESC"
      cursor.execute(
          f"""
        SELECT
          opcode,
          COUNT(*) AS count,
          SUM(bytes_accessed) AS total_bytes,
          SUM(flops) AS total_flops,
          SUM(optimal_seconds) AS total_optimal_seconds
        FROM instructions
        {where_clause}
        GROUP BY opcode
        ORDER BY {order_col}
        LIMIT ?
      """,
          (*params, limit),
      )
      rows = [dict(r) for r in cursor.fetchall()]
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "opcode_stats",
          "rows": rows,
          "opcode_stats": rows,
      }
      self._query_cache[cache_key] = out
      return out

    if mode_clean == "fusion_blockers":
      where_parts = ["is_fusion_blocker = 1"]
      params = []
      if eff_comp:
        where_parts.append("computation_name = ?")
        params.append(eff_comp.lstrip("%"))
      if eff_filter:
        where_parts.append(
            "(name LIKE ? OR expression LIKE ? OR opcode LIKE ?)"
        )
        pat = f"%{eff_filter}%"
        params.extend([pat, pat, pat])
      cursor.execute(
          f"""
        SELECT id, name, computation_name, opcode, shape, layout,
               operand_count, user_count, expression
        FROM instructions
        WHERE {' AND '.join(where_parts)}
        ORDER BY user_count + operand_count DESC, id ASC
        LIMIT ?
      """,
          (*params, limit),
      )
      blockers = []
      for r in cursor.fetchall():
        d = dict(r)
        iid = d["id"]
        u_ids = self.users_by_id.get(iid, [])
        first_consumer_id = u_ids[0] if u_ids else None
        consumer_name = (
            self.id_to_name.get(first_consumer_id, "")
            if first_consumer_id is not None
            else ""
        )
        consumer_opcode = (
            self._get_opcode(first_consumer_id)
            if first_consumer_id is not None
            else ""
        )
        d["blocker_name"] = d["name"]
        d["blocker_opcode"] = d["opcode"]
        d["consumer_name"] = consumer_name
        d["consumer_opcode"] = consumer_opcode
        d["reason"] = get_fusion_blocker_reason(d["opcode"], d["expression"])
        d["producers"] = [
            self.id_to_name.get(oid, str(oid))
            for oid in self.operands_by_id.get(iid, ())
        ]
        d["consumers"] = [self.id_to_name.get(uid, str(uid)) for uid in u_ids]
        blockers.append(d)
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "fusion_blockers",
          "count": len(blockers),
          "fusion_blockers": blockers,
      }
      self._query_cache[cache_key] = out
      return out

    if mode_clean == "layout_transitions":
      cursor.execute(
          """
        SELECT
          src.name AS producer_name,
          src.opcode AS producer_opcode,
          src.shape AS producer_shape,
          src.layout AS producer_layout,
          dst.name AS consumer_name,
          dst.opcode AS consumer_opcode,
          dst.shape AS consumer_shape,
          dst.layout AS consumer_layout,
          dst.computation_name AS computation_name
        FROM edges e
        JOIN instructions src ON e.src_id = src.id
        JOIN instructions dst ON e.dst_id = dst.id
        WHERE
          (src.layout != '' AND dst.layout != '' AND src.layout != dst.layout)
          OR dst.opcode IN ('copy', 'bitcast', 'reshape', 'transpose')
          OR dst.expression LIKE '%AnalyzeLayout%'
        LIMIT ?
      """,
          (limit,),
      )
      transitions = [dict(r) for r in cursor.fetchall()]
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "layout_transitions",
          "count": len(transitions),
          "transitions": transitions,
      }
      self._query_cache[cache_key] = out
      return out

    if mode_clean in ("critical_path", "hotspots"):
      cursor.execute(
          """
        SELECT
          id, name, computation_name, opcode, shape, layout,
          flops, bytes_accessed, optimal_seconds,
          operand_count, user_count, expression
        FROM instructions
        ORDER BY
          optimal_seconds DESC,
          bytes_accessed DESC,
          flops DESC,
          (operand_count + user_count) DESC
        LIMIT ?
      """,
          (limit,),
      )
      hotspots = [dict(r) for r in cursor.fetchall()]
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "critical_path",
          "high_fanout_instructions": hotspots,
          "hotspots": hotspots,
      }
      self._query_cache[cache_key] = out
      return out

    if mode_clean == "call_tree":
      cursor.execute(
          """
        SELECT
          ce.caller_comp_name,
          i.name AS caller_instruction,
          i.opcode AS caller_opcode,
          ce.callee_comp_name,
          c.instruction_count AS callee_instruction_count
        FROM call_edges ce
        JOIN instructions i ON ce.caller_instr_id = i.id
        LEFT JOIN computations c ON ce.callee_comp_name = c.name
        LIMIT ?
      """,
          (limit,),
      )
      calls = [dict(r) for r in cursor.fetchall()]
      out = {
          "status": "SUCCESS",
          "module_name": self.module_name,
          "mode": "call_tree",
          "entry_computation": self.entry_computation_name,
          "call_edges": calls,
      }
      self._query_cache[cache_key] = out
      return out

    # Default / 'search' / 'filter' mode
    where_clauses: list[str] = []
    params = []
    if opcode:
      where_clauses.append("opcode = ?")
      params.append(opcode)
    if eff_comp:
      where_clauses.append("computation_name = ?")
      params.append(eff_comp.lstrip("%"))
    if category:
      where_clauses.append("category LIKE ?")
      params.append(f"%{category}%")
    if eff_filter:
      where_clauses.append(
          "(name LIKE ? OR opcode LIKE ? OR expression LIKE ?"
          " OR metadata_op_name LIKE ?)"
      )
      pat = f"%{eff_filter}%"
      params.extend([pat, pat, pat, pat])

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    cursor.execute(
        f"""
      SELECT
        id, name, computation_name, opcode, shape, layout,
        is_fusion_blocker, operand_count, user_count,
        bytes_accessed, flops, expression
      FROM instructions
      {where_sql}
      ORDER BY id ASC
      LIMIT ?
    """,
        (*params, limit),
    )
    matches = [dict(r) for r in cursor.fetchall()]
    out = {
        "status": "SUCCESS",
        "module_name": self.module_name,
        "mode": mode_clean,
        "count": len(matches),
        "match_count": len(matches),
        "rows": matches,
        "instructions": matches,
    }
    self._query_cache[cache_key] = out
    return out
