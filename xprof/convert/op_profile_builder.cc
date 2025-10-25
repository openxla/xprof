/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/op_profile_builder.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/op_metrics_db_combiner.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"
#include "xprof/utils/op_metrics_db_utils.h"
#include "xprof/utils/xla_op_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using op_profile::Metrics;
using op_profile::Node;
using tsl::profiler::IsFusion;

double CapUtilization(double utilization) { return std::min(utilization, 1.0); }

// Fill symbol details into a node.
void PopulateSymbolNode(const OpMetrics& op_metrics, Node* node) {
  node->set_name(op_metrics.name());
  Node::XLAInstruction& xla = *node->mutable_xla();
  xla.set_program_id(op_metrics.hlo_module_id());
  xla.set_expression(op_metrics.long_name());
  xla.set_xprof_kernel_metadata(
      tensorflow::profiler::ExtractXprofKernelMetadata(op_metrics.long_name()));
  xla.set_fingerprint(op_metrics.fingerprint());
  xla.set_category(op_metrics.category());
  xla.set_provenance(op_metrics.provenance());
  if (op_metrics.has_layout()) {
    for (const auto& dimension : op_metrics.layout().dimensions()) {
      auto* dim = xla.mutable_layout()->add_dimensions();
      dim->set_size(dimension.size());
      dim->set_alignment(dimension.alignment());
      dim->set_semantics(absl::AsciiStrToLower(
          LayoutDimensionSemantics_Name(dimension.semantics())));
    }
  }
  xla.set_computation_primitive_size(op_metrics.computation_primitive_size());
  *xla.mutable_source_info() = op_metrics.source_info();
}

// Sort the children and only keep the top K children.
template <typename Cmp>
Node TopKChildren(const Node* root, int k, Cmp cmp) {
  std::vector<const Node*> children_ptrs;
  children_ptrs.reserve(root->children_size());
  for (const Node& node : root->children()) {
    children_ptrs.push_back(&node);
  }

  // Ensure k is not larger than the number of children
  const int actual_k = std::min(k, static_cast<int>(children_ptrs.size()));

  if (actual_k > 0) {
    std::partial_sort(children_ptrs.begin(), children_ptrs.begin() + actual_k,
                      children_ptrs.end(), cmp);
  }

  Node output;
  for (int i = 0; i < actual_k; ++i) {
    *output.add_children() = *children_ptrs[i];
  }
  return output;
}

// Copy symbol details into a deduplicated node from the top child node.
void CopySymbolDetailsToDeduplicatedNode(Node* top_child_node,
                                         Node* deduplicated_node) {
  deduplicated_node->set_name(
      absl::StrCat(top_child_node->name(), " and its duplicate(s)"));
  Node::XLAInstruction& xla = *deduplicated_node->mutable_xla();
  const Node::XLAInstruction& top_child_node_xla = top_child_node->xla();
  xla.set_program_id(top_child_node_xla.program_id());
  xla.set_expression(top_child_node_xla.expression());
  if (top_child_node_xla.has_xprof_kernel_metadata()) {
    xla.set_xprof_kernel_metadata(top_child_node_xla.xprof_kernel_metadata());
  }
  if (top_child_node_xla.has_source_info()) {
    *xla.mutable_source_info() = top_child_node_xla.source_info();
  }
  xla.set_fingerprint(top_child_node_xla.fingerprint());
  xla.set_category(top_child_node_xla.category());
  if (IsFusion(top_child_node_xla.category())) return;
  xla.set_provenance(top_child_node_xla.provenance());
  *xla.mutable_layout() = top_child_node_xla.layout();
}

bool RequiresProgramGrouping(OpProfileOptions& options) {
  return options.group_by == OpProfileGrouping::kByProgram ||
         options.group_by == OpProfileGrouping::kByProvenance;
}

void SortAndPruneChildren(int k, int level, Node* root) {
  // Set the total number of children before pruning.
  root->set_num_children(root->children_size());
  for (Node& node : *root->mutable_children()) {
    SortAndPruneChildren(k, level - 1, &node);
  }
  k = level > 0 ? root->children_size() : k;

  if (root->children_size() > 1) {
    if (root->has_xla() && IsFusion(root->xla().category())) {
      // Sort the children under fusion node by raw flops.
      *root->mutable_children() =
          TopKChildren(root, k, [](const Node* a, const Node* b) {
            return a->metrics().raw_flops() > b->metrics().raw_flops();
          }).children();
    } else {
      *root->mutable_children() =
          TopKChildren(root, k, [](const Node* a, const Node* b) {
            return a->metrics().raw_time() > b->metrics().raw_time();
          }).children();
    }
  }
}

void FinalizeNodeRecursive(Node* node) {
  // A node is a deduplication group if it has children, it does NOT have XLA
  // info, and ALL of its children DO have XLA info.
  bool is_dedup_group = (node->children_size() > 0 && !node->has_xla());
  if (is_dedup_group) {
    for (const auto& child : node->children()) {
      if (!child.has_xla()) {
        is_dedup_group = false;
        break;
      }
    }
  }

  // If it's not a dedup group, recurse into its children.
  if (!is_dedup_group) {
    for (int i = 0; i < node->children_size(); ++i) {
      FinalizeNodeRecursive(node->mutable_children(i));
    }
    return;
  }

  // It's a single-op group. Replace the node with its child.
  // A temporary copy is used to avoid the protobuf descendant check error.
  if (node->children_size() == 1) {
    Node child_copy = *node->mutable_children(0);
    *node = child_copy;
    return;  // Node is replaced; no need to recurse further.
  }

  // It's a multi-op group. Finalize its name.
  // Children are op-nodes; no need to recurse further.
  CopySymbolDetailsToDeduplicatedNode(node->mutable_children(0), node);
}

void FinalizeDeduplicatedNodes(Node* root) {
  if (root) {
    FinalizeNodeRecursive(root);
  }
}

// Fills op metrics into a node.
void PopulateOpMetricsNode(
    const OpMetrics& op_metrics, double peak_gigaflops_per_second_per_core,
    std::vector<double> peak_mem_gibibytes_per_second_per_core,
    uint64_t total_time_ps, Node* node) {
  Metrics* metrics = node->mutable_metrics();
  // The UI computes flops_rate = raw_flops / raw_time
  // and memory_bandwidth = raw_bytes_accessed / raw_time. See:
  // https://github.com/tensorflow/profiler/blob/master/frontend/app/common/utils/utils.ts
  metrics->set_raw_time(op_metrics.time_ps());
  metrics->set_raw_flops(op_metrics.model_flops());
  metrics->set_bf16_flops(op_metrics.flops());
  metrics->set_occurrences(op_metrics.occurrences());
  metrics->set_avg_time_ps(tsl::profiler::SafeDivide(op_metrics.time_ps(),
                                                     op_metrics.occurrences()));

  double uncapped_flops_utilization =
      tsl::profiler::SafeDivide(GigaFlopsPerSecondPerCore(op_metrics),
                                peak_gigaflops_per_second_per_core);

  double flops_utilization = CapUtilization(uncapped_flops_utilization);
  // The UI expects flops_utilization = flop_util / time_fraction. See:
  // https://github.com/tensorflow/profiler/blob/master/frontend/app/common/utils/utils.ts
  const double time_fraction =
      tsl::profiler::SafeDivide(op_metrics.time_ps(), total_time_ps);
  metrics->set_flops(flops_utilization * time_fraction);
  metrics->set_uncapped_flops(uncapped_flops_utilization * time_fraction);
  metrics->set_normalized_time_ps(op_metrics.normalized_time_ps());

  // Capture both on-chip and off-chip memory utilization.
  const double hbm_gibibytes_per_second =
      tsl::profiler::GigaToGibi(
          GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_HBM,
                                    OpMetrics::MemoryAccessed::READ)) +
      tsl::profiler::GigaToGibi(
          GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_HBM,
                                    OpMetrics::MemoryAccessed::WRITE));
  const double hbm_bw_utilization = CapUtilization(tsl::profiler::SafeDivide(
      hbm_gibibytes_per_second,
      peak_mem_gibibytes_per_second_per_core[MemBwType::MEM_BW_TYPE_HBM_RW]));
  metrics->add_bandwidth_utils(hbm_bw_utilization);
  double hbm_bytes = tsl::profiler::GibiToGiga(hbm_gibibytes_per_second) *
                     tsl::profiler::PicoToNano(op_metrics.time_ps());

  const double sram_rd_gibibytes_per_second = tsl::profiler::GigaToGibi(
      GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_ON_CHIP,
                                OpMetrics::MemoryAccessed::READ));
  const double sram_rd_bw_utilization =
      CapUtilization(tsl::profiler::SafeDivide(
          sram_rd_gibibytes_per_second, peak_mem_gibibytes_per_second_per_core
                                            [MemBwType::MEM_BW_TYPE_SRAM_RD]));
  metrics->add_bandwidth_utils(sram_rd_bw_utilization);
  double sram_rd_bytes =
      tsl::profiler::GibiToGiga(sram_rd_gibibytes_per_second) *
      tsl::profiler::PicoToNano(op_metrics.time_ps());

  const double sram_wr_gibibytes_per_second = tsl::profiler::GigaToGibi(
      GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_ON_CHIP,
                                OpMetrics::MemoryAccessed::WRITE));
  const double sram_wr_bw_utilization =
      CapUtilization(tsl::profiler::SafeDivide(
          sram_wr_gibibytes_per_second, peak_mem_gibibytes_per_second_per_core
                                            [MemBwType::MEM_BW_TYPE_SRAM_WR]));
  metrics->add_bandwidth_utils(sram_wr_bw_utilization);
  double sram_wr_bytes =
      tsl::profiler::GibiToGiga(sram_wr_gibibytes_per_second) *
      tsl::profiler::PicoToNano(op_metrics.time_ps());

  metrics->add_raw_bytes_accessed_array(hbm_bytes);
  metrics->add_raw_bytes_accessed_array(sram_rd_bytes);
  metrics->add_raw_bytes_accessed_array(sram_wr_bytes);
}

// Recursively insert "fused instruction" nodes (with raw flops).
void InsertFusedInstructions(const OpMetrics& op_metrics, Node* node) {
  if (!op_metrics.has_children()) return;
  for (const auto& child : op_metrics.children().metrics_db()) {
    Node* new_node = node->add_children();
    PopulateSymbolNode(child, new_node);
    new_node->mutable_metrics()->set_raw_flops(child.model_flops());
    new_node->mutable_metrics()->set_bf16_flops(child.flops());
    if (child.has_children()) {
      InsertFusedInstructions(child, new_node);
    }
  }
}

// Update child node metrics to parent node.
void UpdateNodeMetrics(const OpMetrics& child, OpMetrics* parent) {
  DCHECK(parent != nullptr);
  parent->set_time_ps(child.self_time_ps() + parent->time_ps());
  parent->set_normalized_time_ps(child.normalized_time_ps() +
                                 parent->normalized_time_ps());
  parent->set_self_time_ps(child.self_time_ps() + parent->self_time_ps());
  if (ChildrenTimePs(child) == 0) {
    parent->set_flops(child.flops() + parent->flops());
    parent->set_model_flops(child.model_flops() + parent->model_flops());
    parent->set_bytes_accessed(child.bytes_accessed() +
                               parent->bytes_accessed());
    parent->set_dma_stall_ps(child.dma_stall_ps() + parent->dma_stall_ps());
    CombineMemoryAccessedBreakdown(child.memory_accessed_breakdown(),
                                   parent->mutable_memory_accessed_breakdown());
  }
}

}  // namespace

std::string OpProfileBuilder::GenerateProgramName(uint64_t program_id) const {
  DCHECK(program_name_map_ != nullptr);
  auto iter = program_name_map_->find(program_id);
  if (iter == program_name_map_->end()) return "main";
  return tsl::profiler::HloModuleNameWithProgramId(iter->second, program_id);
}

Node* OpProfileBuilder::AddOpNode(const OpMetrics& op_metrics,
                                  Category* category, Node* deduplicated_node) {
  Node* leaf;
  if (deduplicated_node != nullptr) {
    leaf = deduplicated_node->add_children();
  } else if (category != nullptr) {
    leaf = category->node->add_children();
  } else {
    leaf = root_->add_children();
  }
  PopulateSymbolNode(op_metrics, leaf);
  InsertFusedInstructions(op_metrics, leaf);
  return leaf;
}

// Function to create deduplicated aggregation layer.
// 1. Empty deduplicated_name in op_metrics means either:
// (1) a grouping op of a deduplicated op list. (fusion.3 in the example below)
// (2) an op that does not have duplicates. (fusion.4 in the example below)
// We create dedup layer for both cases due to lack of clue which case it is.
// The op name is used directly as the hash key for the dedup group. The dedup
// layer will be removed in the 2nd pass for case (2).
// 2. Non-empty deduplicated_name means this op can be grouped to a
// deduplicated op list (fusion.1 in the example below).
// Example:
// op_metrics {
//   name: "fusion.1"
//   deduplicated_name: "fusion.3"
//   category: "convolution"
// }
// op_metrics {
//   name: "fusion.3"
//   deduplicated_name: ""
//   category: "convolution"
// }
// op_metrics {
//   name: "fusion.4"
//   deduplicated_name: ""
//   category: "convolution"
// }
// The data above will create the following tree after calling the function
// repeatedly:
// root(by_program)
// - jit.xx
//   - convolution
//     - fusion.3
//       - fusion.1
//       - fusion.2
//       - fusion.3
//     - fusion.4
//       - fusion.4
// After finalization, the tree will look like:
// root(by_program)
// - jit.xx
//   - convolution
//     - fusion.3 and its duplicate(s)
//       - fusion.1
//       - fusion.2
//       - fusion.3
//     - fusion.4
Node* OpProfileBuilder::LookupOrAddDeduplicatedNode(const OpMetrics& op_metrics,
                                                    Category* category) {
  std::string deduplicated_name = op_metrics.deduplicated_name().empty()
                                      ? op_metrics.name()
                                      : op_metrics.deduplicated_name();
  Node*& deduplicated_node = category->deduplicated_nodes[deduplicated_name];
  if (deduplicated_node == nullptr) {
    deduplicated_node = category->node->add_children();
    // Set deduplicated name which is the hash key for the dedup group.
    // Symbol details will be added in finalization step.
    deduplicated_node->set_name(deduplicated_name);
  }
  return deduplicated_node;
}

OpProfileBuilder::Category* OpProfileBuilder::LookupOrAddCategoryNode(
    const OpMetrics& op_metrics, Program* program,
    op_profile::Node* provenance_leaf_node) {
  Category* category;
  Node* category_parent;

  if (provenance_leaf_node != nullptr) {
    // Grouping by provenance. Parent is the leaf of the provenance path.
    category_parent = provenance_leaf_node;
    category =
        &provenance_category_map_[category_parent][op_metrics.category()];
  } else if (program != nullptr) {
    // Grouping by program. Parent is the program node.
    category = &program->categories[op_metrics.category()];
    category_parent = program->node;
  } else {
    // Grouping by category only. Parent is the root node.
    category = &category_map_[op_metrics.category()];
    category_parent = root_;
  }

  if (category->node == nullptr) {
    category->node = category_parent->add_children();
    category->node->set_name(op_metrics.category());
  }
  return category;
}

OpProfileBuilder::Program* OpProfileBuilder::LookupOrAddProgramNode(
    const OpMetrics& op_metrics) {
  uint64_t program_id = op_metrics.hlo_module_id();
  Program* program = &programs_map_[program_id];
  if (program->node == nullptr) {
    program->node = root_->add_children();
    program->node->set_name(GenerateProgramName(program_id));
  }
  return program;
}

Node* OpProfileBuilder::GetOrAddProvenanceParentNode(
    const OpMetrics& op_metrics, Program* program) {
  Node* current_node = nullptr;
  if (program != nullptr) {
    current_node = program->node;
  } else {
    current_node = root_;
  }

  std::vector<std::string> provenance_parts =
      tensorflow::profiler::ParseProvenance(op_metrics.provenance());
  for (const auto& name : provenance_parts) {
    auto& children = provenance_children_map_[current_node];
    Node*& child_node = children[name];
    if (child_node == nullptr) {
      child_node = current_node->add_children();
      child_node->set_name(name);
    }
    UpdateNodeMetrics(op_metrics, &metrics_[child_node]);
    current_node = child_node;
  }
  return current_node;
}

void OpProfileBuilder::AddOp(const OpMetrics& op_metrics) {
  // 1. Deal with nested parent nodes (vertical grouping)
  // op_metrics.time_ps in root node will be reset to total_time_ps later
  UpdateNodeMetrics(op_metrics, &metrics_[root_]);
  Program* program = nullptr;
  if (!IsIdleOp(op_metrics) && RequiresProgramGrouping(options_)) {
    program = LookupOrAddProgramNode(op_metrics);
    UpdateNodeMetrics(op_metrics, &metrics_[program->node]);
  }

  // 2. Deal with nested grouping nodes (horizontal grouping), only accumulate
  // non-child ops Exclude ops with children ops to avoid double counting of
  // flops, bytes and time from children ops.
  if (ChildrenTimePs(op_metrics) > 0) return;
  std::vector<Node*> nested_grouping_nodes;
  if (IsIdleOp(op_metrics)) {
    Node* leaf = AddOpNode(op_metrics);
    nested_grouping_nodes.push_back(leaf);
  } else {
    op_profile::Node* provenance_leaf_node = nullptr;
    if ((options_.group_by == OpProfileGrouping::kByProvenance) &&
        !op_metrics.provenance().empty()) {
      provenance_leaf_node = GetOrAddProvenanceParentNode(op_metrics, program);
    }

    Category* category =
        LookupOrAddCategoryNode(op_metrics, program, provenance_leaf_node);
    nested_grouping_nodes.push_back(category->node);

    Node* deduplicated_node = nullptr;
    if (options_.group_by_deduplicated_name) {
      deduplicated_node = LookupOrAddDeduplicatedNode(op_metrics, category);
      nested_grouping_nodes.push_back(deduplicated_node);
    }

    Node* leaf = AddOpNode(op_metrics, category, deduplicated_node);
    nested_grouping_nodes.push_back(leaf);
  }

  for (auto* node : nested_grouping_nodes) {
    // Per program combiner does not need to update OpMetrics.num_cores
    CombineOpMetrics(op_metrics, &metrics_[node], /*update_num_cores=*/false);
  }
}

void OpProfileBuilder::Finalize(
    double peak_gigaflops_per_second_per_core,
    std::vector<double> peak_mem_gibibytes_per_second_per_core,
    uint64_t total_time_ps) {
  // Call to `PopulateOpMetricsNode` depends on node time_ps to calculate
  // flops, bandwidth_utils..etc. The root / program node time_ps might
  // be off a bit, missing its own self_time when calling `UpdateNodeMetrics`.
  // This is best effort to at least reset the time_ps for root node to be more
  // precise.
  metrics_[root_].set_time_ps(total_time_ps);
  for (const auto& [node, op_metrics] : metrics_) {
    PopulateOpMetricsNode(op_metrics, peak_gigaflops_per_second_per_core,
                          peak_mem_gibibytes_per_second_per_core, total_time_ps,
                          node);
  }
  // If grouping by program, we build a two-level pruned tree: the first level
  // is per program and the second level is per category. Otherwise we build a
  // single-level per category pruned tree.
  // for kByProvenance, we build a dynamic-depth pruned tree with a large
  // level.
  int level = 2;
  switch (options_.group_by) {
    case OpProfileGrouping::kByProgram:
      level = 2;
      break;
    case OpProfileGrouping::kByCategory:
      level = 1;
      break;
    case OpProfileGrouping::kByProvenance:
      // A large number prevents pruning of the
      //  dynamic-depth provenance hierarchy.
      level = 100;
      break;
  }
  SortAndPruneChildren(options_.children_per_node, level, root_);
  SortAndPruneChildren(options_.children_per_node, level, root_);
  if (options_.group_by_deduplicated_name) {
    FinalizeDeduplicatedNodes(root_);
  }
}

OpProfileBuilder::OpProfileBuilder(
    const OpProfileOptions& options,
    tensorflow::profiler::op_profile::Node* root,
    const tsl::protobuf::Map<uint64_t, std::string>* program_name_map)
    : options_(options), root_(root), program_name_map_(program_name_map) {
  if (root == nullptr) {
    LOG(DFATAL) << "root is null.";
    return;
  }
  // Both kByProgram and kByProvenance require a program_name_map.
  if (RequiresProgramGrouping(options_)) {
    DCHECK(program_name_map_ != nullptr)
        << "program_name_map is required for grouping by program or "
           "provenance.";
  }
  // Set the root node's name based on the selected grouping option.
  switch (options_.group_by) {
    case OpProfileGrouping::kByProgram:
      root->set_name("by_program");
      break;
    case OpProfileGrouping::kByProvenance:
      root->set_name("by_provenance");
      break;
    case OpProfileGrouping::kByCategory:
      root->set_name("by_category");
      break;
  }
}

}  // namespace profiler
}  // namespace tensorflow
