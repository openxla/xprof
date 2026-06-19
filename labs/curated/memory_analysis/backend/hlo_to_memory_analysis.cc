/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xprof/labs/curated/memory_analysis/backend/hlo_to_memory_analysis.h"

#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"  // from @com_github_nlohmann_json
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/hlo_proto_to_memory_visualization_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

struct TraceState {
  const xla::HloInstruction* inst;
  const xla::HloInstruction* calling_fusion;
};

bool IsDerivedFromWeight(
    const xla::HloInstruction* inst,
    absl::flat_hash_set<const xla::HloInstruction*>& visited) {
  if (inst->opcode() == xla::HloOpcode::kParameter ||
      inst->opcode() == xla::HloOpcode::kConstant) {
    return true;
  }
  if (!visited.insert(inst).second) {
    return true;
  }

  bool is_allowed = inst->IsElementwise() ||
                    inst->opcode() == xla::HloOpcode::kReshape ||
                    inst->opcode() == xla::HloOpcode::kTranspose ||
                    inst->opcode() == xla::HloOpcode::kBitcast;

  if (!is_allowed) {
    return false;
  }

  return absl::c_all_of(
      inst->operands(), [&visited](const xla::HloInstruction* operand) {
        return IsDerivedFromWeight(operand, visited);
      });
}

// Extracts clean JAX path from TF Op Name
std::string ExtractJaxVariablePath(absl::string_view tf_op) {
  if (tf_op.empty()) return "";

  static const RE2 kFlaxParamsPattern(
      R"(variables\['params'\]((?:\['[^']+'\])+))");
  std::string path_str;
  if (RE2::PartialMatch(tf_op, kFlaxParamsPattern, &path_str)) {
    return absl::StrCat("params", path_str);
  }

  std::vector<absl::string_view> parts = absl::StrSplit(tf_op, '/');

  auto shard_map_it = absl::c_find(parts, "shard_map");
  if (shard_map_it != parts.end()) {
    return absl::StrJoin(shard_map_it + 1, parts.end(), "/");
  }

  static const RE2 kFuncPattern(R"([a-z0-9]+\(.*\))");
  auto func_it = absl::c_find_if(parts, [](absl::string_view part) {
    return RE2::FullMatch(part, kFuncPattern);
  });
  if (func_it != parts.end()) {
    return absl::StrJoin(func_it + 1, parts.end(), "/");
  }

  if (parts.size() > 3) {
    return absl::StrJoin(parts.end() - 3, parts.end(), "/");
  }

  return std::string(tf_op);
}

// Categorize peak heap buffer using heuristics.
std::pair<std::string, std::string> CategorizeTensor(
    absl::string_view tf_op, absl::string_view group,
    absl::string_view shape_string, absl::string_view label,
    const absl::flat_hash_set<std::string>& gathered_weights) {
  std::string category = "Other";
  std::string sub_cat;

  // 1. Graph-based heuristic for gathered weights
  std::vector<absl::string_view> label_parts = absl::StrSplit(label, ':');
  if (!label_parts.empty()) {
    if (gathered_weights.contains(absl::StripAsciiWhitespace(label_parts[0]))) {
      return {"Weight", "Gathered"};
    }
  }

  // 2. SparseCore based on layout L(1024) combined with Group
  if (absl::StrContains(shape_string, "L(1024)")) {
    if (group == "Parameter") {
      category = "Input";
      sub_cat = "SparseCore";
    } else {
      category = "SparseCore";
      sub_cat = "Pipeline";
    }
  }

  // Check for specific keywords avoiding top-level function name
  std::vector<absl::string_view> parts = absl::StrSplit(tf_op, '/');
  std::string rest;
  if (parts.size() > 1) {
    rest = absl::StrJoin(parts.begin() + 1, parts.end(), "/");
  } else {
    rest = std::string(tf_op);
  }

  if (absl::StrContains(rest, "embedding_activations")) {
    category = "SparseCore";
    sub_cat = "Activation";
  } else if (absl::StrContains(rest, "embedding_gradients")) {
    category = "SparseCore";
    sub_cat = "Gradient";
  } else if (absl::StrContains(rest, "embedding") ||
             absl::StrContains(rest, "sparsecore")) {
    category = "SparseCore";
    if (sub_cat.empty()) {
      sub_cat = "Other";
    }
  }

  // 3. New heuristic for "TC format"
  if (category == "Other" && group == "Output" &&
      absl::StrContains(tf_op, "pipelined_tc") &&
      absl::StrContains(tf_op, "reshape")) {
    static const RE2 kTcShapePattern(R"(,\s*(168|96)\])");
    if (RE2::PartialMatch(shape_string, kTcShapePattern)) {
      category = "SparseCore";
      sub_cat = "TC format";
    }
  }

  // 4. Differentiate based on Group to avoid false positives
  if (category == "Other") {
    if (group == "Parameter") {
      if (absl::StrContains(tf_op, "param") ||
          absl::StrContains(tf_op, "weight") ||
          absl::StrContains(tf_op, "variables['params']")) {
        category = "Weight";
      } else if (absl::StrContains(tf_op, "input") ||
                 absl::StrContains(tf_op, "feature") ||
                 absl::StrContains(tf_op, "batch")) {
        category = "Input";
      } else if (absl::StrContains(tf_op, "optimizer") ||
                 absl::StrContains(tf_op, "opt_state")) {
        category = "Optimizer State";
      } else {
        category = "Uncategorized Parameter";
      }
    } else if (group == "Temporary") {
      if (absl::StrContains(tf_op, "Gradient") ||
          absl::StrContains(tf_op, "grad")) {
        category = "Gradient";
      } else if (absl::StrContains(tf_op, "optimizer") ||
                 absl::StrContains(tf_op, "Adam") ||
                 absl::StrContains(tf_op, "opt")) {
        category = "Optimizer State";
      } else if (absl::StrContains(tf_op, "param") ||
                 absl::StrContains(tf_op, "weight") ||
                 absl::StrContains(tf_op, "variables['params']")) {
        if (!absl::StrContains(tf_op, "dot_general") &&
            !absl::StrContains(tf_op, "fusion") &&
            !absl::StrContains(tf_op, "convolution")) {
          category = "Weight";
          sub_cat = "Gathered";
        } else {
          category = "Temporary/Activation";
        }
      } else {
        category = "Temporary/Activation";
      }
    } else if (group == "Output") {
      category = "Output";
    }
  }

  return {category, sub_cat};
}

std::string GetMemorySpaceName(int64_t color) {
  switch (color) {
    case 0:
      return "HBM";
    case 1:
      return "VMem";
    case 2:
      return "Sflag";
    case 3:
      return "CMem";
    case 4:
      return "SMem";
    case 5:
      return "System Memory";
    case 6:
      return "Workspace Register";
    case 7:
      return "SparseCore Private Stack HBM";
    case 8:
      return "SparseCore Sequencer Sflag";
    case 9:
      return "Pinned HBM";
    case 10:
      return "SparseCore Tile Sflag";
    default:
      return absl::StrCat("Memory Space ", color);
  }
}

}  // namespace

absl::StatusOr<absl::flat_hash_set<std::string>> TraceGatheredWeights(
    const xla::HloProto& hlo_proto) {
  if (!hlo_proto.hlo_module().has_host_program_shape()) {
    return absl::InvalidArgumentError("No program shape found in the proto");
  }
  TF_ASSIGN_OR_RETURN(xla::ProgramShape program_shape,
                      xla::ProgramShape::FromProto(
                          hlo_proto.hlo_module().host_program_shape()));
  xla::HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> module,
      xla::HloModule::CreateFromProto(hlo_proto.hlo_module(), config));

  absl::flat_hash_set<const xla::HloInstruction*> gathered_weights;
  absl::flat_hash_set<const xla::HloInstruction*> visited;
  std::queue<TraceState> q;

  for (const xla::HloComputation* computation : module->computations()) {
    for (const xla::HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == xla::HloOpcode::kAllGather) {
        absl::flat_hash_set<const xla::HloInstruction*> backward_visited;
        if (IsDerivedFromWeight(inst->operand(0), backward_visited)) {
          q.push({inst, nullptr});
          visited.insert(inst);
        }
      }
    }
  }

  while (!q.empty()) {
    TraceState state = q.front();
    q.pop();
    const xla::HloInstruction* inst = state.inst;
    const xla::HloInstruction* calling_fusion = state.calling_fusion;

    bool is_allowed = inst->IsElementwise() ||
                      inst->opcode() == xla::HloOpcode::kReshape ||
                      inst->opcode() == xla::HloOpcode::kTranspose ||
                      inst->opcode() == xla::HloOpcode::kBitcast ||
                      inst->opcode() == xla::HloOpcode::kAllGather ||
                      inst->opcode() == xla::HloOpcode::kParameter ||
                      inst->opcode() == xla::HloOpcode::kGetTupleElement ||
                      inst->opcode() == xla::HloOpcode::kTuple ||
                      inst->opcode() == xla::HloOpcode::kBroadcast;

    if (!is_allowed) {
      continue;
    }

    gathered_weights.insert(inst);

    if (inst == inst->parent()->root_instruction() &&
        calling_fusion != nullptr) {
      gathered_weights.insert(calling_fusion);
      for (const xla::HloInstruction* user : calling_fusion->users()) {
        if (visited.insert(user).second) {
          q.push({user, nullptr});
        }
      }
      continue;
    }

    for (const xla::HloInstruction* user : inst->users()) {
      if (user->opcode() == xla::HloOpcode::kFusion) {
        const xla::HloComputation* called_comp = user->called_computations()[0];
        for (int i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) == inst) {
            const xla::HloInstruction* param =
                called_comp->parameter_instruction(i);
            if (visited.insert(param).second) {
              q.push({param, user});
            }
          }
        }
      } else {
        if (visited.insert(user).second) {
          q.push({user, calling_fusion});
        }
      }
    }
  }

  absl::flat_hash_set<std::string> gathered_weight_names;
  for (const xla::HloInstruction* inst : gathered_weights) {
    gathered_weight_names.insert(std::string(inst->name()));
  }

  return gathered_weight_names;
}

absl::StatusOr<std::string> ConvertHloProtoToMemoryAnalysisJson(
    const xla::HloProto& hlo_proto, absl::string_view module_name) {
  absl::flat_hash_set<std::string> gathered_weights;
  if (absl::StatusOr<absl::flat_hash_set<std::string>> weights =
          TraceGatheredWeights(hlo_proto);
      weights.ok()) {
    gathered_weights = *std::move(weights);
  } else {
    LOG(WARNING) << "Failed to trace gathered weights: "
                 << weights.status().message();
  }

  absl::flat_hash_set<int64_t> memory_colors;
  if (hlo_proto.has_buffer_assignment()) {
    for (const xla::BufferAllocationProto& allocation :
         hlo_proto.buffer_assignment().buffer_allocations()) {
      memory_colors.insert(allocation.color());
    }
  }
  if (memory_colors.empty()) {
    memory_colors.insert(0);
  }

  nlohmann::json final_json;
  final_json["moduleName"] = std::string(module_name);

  nlohmann::json summary;
  summary["Weight"] = 0;
  summary["Input"] = 0;
  summary["Optimizer State"] = 0;
  summary["SparseCore"] = 0;
  summary["Temporary/Activation"] = 0;
  summary["Output"] = 0;
  summary["Other"] = 0;
  summary["Uncategorized Parameter"] = 0;
  summary["Gradient"] = 0;

  nlohmann::json memory_space_breakdown = nlohmann::json::object();
  nlohmann::json peak_heap_buffers = nlohmann::json::array();

  for (int64_t color : memory_colors) {
    std::string space_name = GetMemorySpaceName(color);

    MemoryViewerOption option;
    option.memory_color = color;

    TF_ASSIGN_OR_RETURN(
        std::string preprocessed_json_str,
        ConvertHloProtoToPreprocessResultJson(hlo_proto, option));
    nlohmann::json preprocessed_json =
        nlohmann::json::parse(preprocessed_json_str);

    double peak_heap_mib = preprocessed_json.value("peakHeapMib", 0.0);
    int64_t peak_bytes = static_cast<int64_t>(peak_heap_mib * (1ULL << 20));
    memory_space_breakdown[space_name] = peak_bytes;

    if (preprocessed_json.contains("maxHeap")) {
      for (const nlohmann::json& buffer : preprocessed_json["maxHeap"]) {
        std::string label = buffer.value("label", "");
        double size_mib = buffer.value("logicalBufferSizeMib", 0.0);
        double unpadded_size_mib = buffer.value("unpaddedShapeMib", size_mib);
        int64_t size_bytes = static_cast<int64_t>(size_mib * (1ULL << 20));
        int64_t unpadded_bytes =
            static_cast<int64_t>(unpadded_size_mib * (1ULL << 20));
        std::string group = buffer.value("groupName", "");
        std::string tf_op = buffer.value("tfOpName", "");
        std::string shape = buffer.value("shapeString", "");

        auto [category, sub_category] =
            CategorizeTensor(tf_op, group, shape, label, gathered_weights);

        summary[category] = summary.value(category, 0LL) + size_bytes;

        nlohmann::json buf_json;
        buf_json["tfOp"] = tf_op;
        buf_json["jaxPath"] = ExtractJaxVariablePath(tf_op);
        buf_json["group"] = group;
        buf_json["size"] = size_bytes;
        buf_json["unpaddedSize"] = unpadded_bytes;
        buf_json["shape"] = shape;
        buf_json["label"] = label;
        buf_json["category"] = category;
        buf_json["subCategory"] = sub_category;
        buf_json["memorySpace"] = space_name;

        peak_heap_buffers.push_back(buf_json);
      }
    }
  }

  final_json["summary"] = summary;
  final_json["memorySpaceBreakdown"] = memory_space_breakdown;
  final_json["peakHeapBuffers"] = peak_heap_buffers;

  return final_json.dump();
}

}  // namespace profiler
}  // namespace tensorflow
