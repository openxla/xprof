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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/nullability.h"
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
#include "xprof/utils/flat_op_metrics_db_utils.h"
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
  if (!xla.source_info().has_stack_frame()) {
    xla.mutable_source_info()->set_stack_frame("");
  }
}

// Fill symbol details into a node for FlatOpMetrics.
void PopulateSymbolNode(const FlatOpMetrics& op_metrics, Node* node) {
  node->set_name(op_metrics.hlo_name());
  Node::XLAInstruction& xla = *node->mutable_xla();
  xla.set_program_id(op_metrics.hlo_module_id());
  xla.set_expression(op_metrics.long_name());
  xla.set_xprof_kernel_metadata(
      tensorflow::profiler::ExtractXprofKernelMetadata(op_metrics.long_name()));
  xla.set_category(op_metrics.category());
  xla.set_provenance(op_metrics.provenance());
  *xla.mutable_source_info() = op_metrics.source_info();
  if (!xla.source_info().has_stack_frame()) {
    xla.mutable_source_info()->set_stack_frame("");
  }
}

// Sorts the children in place and only keeps the top K children.
//
// The children are ordered by running `std::partial_sort` over an index
// permutation and then reordering the repeated field with pointer swaps, so no
// child subtree is copied.
template <typename Cmp>
void TopKChildrenInPlace(Node* root, int k, Cmp cmp) {
  const int n = root->children_size();
  // Ensure k is not larger than the number of children.
  const int actual_k = std::min(k, n);
  if (actual_k <= 0) {
    root->clear_children();
    return;
  }

  // Sort {node, index} entries rather than plain indices: libc++ uses a
  // different heap construction for arithmetic value types, which would order
  // ties differently from sorting `const Node*`.
  struct Entry {
    const Node* node;
    int index;
  };
  std::vector<Entry> order;
  order.reserve(n);
  for (int i = 0; i < n; ++i) order.push_back({&root->children(i), i});
  std::partial_sort(
      order.begin(), order.begin() + actual_k, order.end(),
      [&cmp](const Entry& a, const Entry& b) { return cmp(a.node, b.node); });

  // Apply the permutation with pointer swaps. `position[i]` is the current
  // position of the original child `i`; `original[p]` is the original index of
  // the child currently at position `p`.
  std::vector<int> position(n);
  std::vector<int> original(n);
  for (int i = 0; i < n; ++i) position[i] = original[i] = i;
  auto* mutable_children = root->mutable_children();
  for (int p = 0; p < actual_k; ++p) {
    const int want = order[p].index;
    const int from = position[want];
    if (from == p) continue;
    mutable_children->SwapElements(p, from);
    const int displaced = original[p];
    original[from] = displaced;
    position[displaced] = from;
    original[p] = want;
    position[want] = p;
  }
  if (actual_k < n) {
    mutable_children->DeleteSubrange(actual_k, n - actual_k);
  }
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

template <typename Cmp>
void SortAndPruneChildrenOfNode(int k, Node* root, Cmp cmp) {
  if (root->children_size() > 1) {
    TopKChildrenInPlace(root, k, cmp);
  }
}

// Sorts and prunes the children of `root` recursively. The sort order of a
// node's children depends only on the children's own metrics, so each node is
// pruned before recursing, which skips subtrees that are dropped anyway.
void SortAndPruneChildren(
    int k, int level, Node* root,
    absl::flat_hash_map<Node*, const OpMetrics*>* leaf_op_metrics) {
  if (auto it = leaf_op_metrics->find(root); it != leaf_op_metrics->end()) {
    const OpMetrics& op_metrics = *it->second;
    PopulateSymbolNode(op_metrics, root);
    if (op_metrics.has_children()) {
      for (const auto& child : op_metrics.children().metrics_db()) {
        Node* new_node = root->add_children();
        Metrics* m = new_node->mutable_metrics();
        m->set_raw_time(child.time_ps());
        m->set_raw_flops(child.model_flops_v2());
        (*leaf_op_metrics)[new_node] = &child;
      }
    }
  }
  // `num_children` holds the child count from before pruning; the frontend
  // derives the number of left-out children from it.
  root->set_num_children(root->children_size());
  const int node_k = level > 0 ? root->children_size() : k;
  if (root->has_xla() && IsFusion(root->xla().category())) {
    // Sort the children under fusion node by raw flops.
    SortAndPruneChildrenOfNode(node_k, root, [](const Node* a, const Node* b) {
      return a->metrics().raw_flops() > b->metrics().raw_flops();
    });
  } else {
    SortAndPruneChildrenOfNode(node_k, root, [](const Node* a, const Node* b) {
      return a->metrics().raw_time() > b->metrics().raw_time();
    });
  }
  for (Node& node : *root->mutable_children()) {
    SortAndPruneChildren(k, level - 1, &node, leaf_op_metrics);
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
  // The child is first swapped into a temporary on the same arena (a pointer
  // swap, not a copy) to avoid the protobuf descendant check error.
  if (node->children_size() == 1) {
    tsl::protobuf::Arena* arena = node->GetArena();
    std::unique_ptr<Node> owned_tmp;
    Node* tmp;
    if (arena == nullptr) {
      owned_tmp = std::make_unique<Node>();
      tmp = owned_tmp.get();
    } else {
      tmp = tsl::protobuf::Arena::Create<Node>(arena);
    }
    tmp->Swap(node->mutable_children(0));
    node->Swap(tmp);
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
  // TODO(b/483381236): Remove in favor of total_flops.
  metrics->set_raw_flops(op_metrics.model_flops_v2());
  metrics->set_bf16_flops(op_metrics.flops_v2());
  metrics->set_occurrences(op_metrics.occurrences());
  metrics->set_avg_time_ps(tsl::profiler::SafeDivide(op_metrics.time_ps(),
                                                     op_metrics.occurrences()));

  double uncapped_flops_utilization = tsl::profiler::SafeDivide(
      GigaFlopsPerSecondPerCoreNormalizedOnDvfs(op_metrics),
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
  auto get_peak_mem_bw = [&](size_t index) -> double {
    if (index < peak_mem_gibibytes_per_second_per_core.size()) {
      return peak_mem_gibibytes_per_second_per_core[index];
    }
    return 0.0;
  };

  const double hbm_gibibytes_per_second =
      tsl::profiler::GigaToGibi(
          GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_HBM,
                                    OpMetrics::MemoryAccessed::READ)) +
      tsl::profiler::GigaToGibi(
          GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_HBM,
                                    OpMetrics::MemoryAccessed::WRITE));
  const double hbm_bw_utilization = CapUtilization(tsl::profiler::SafeDivide(
      hbm_gibibytes_per_second,
      get_peak_mem_bw(MemBwType::MEM_BW_TYPE_HBM_RW)));
  metrics->add_bandwidth_utils(hbm_bw_utilization);
  double hbm_bytes = tsl::profiler::GibiToGiga(hbm_gibibytes_per_second) *
                     tsl::profiler::PicoToNano(op_metrics.time_ps());

  const double sram_rd_gibibytes_per_second = tsl::profiler::GigaToGibi(
      GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_ON_CHIP,
                                OpMetrics::MemoryAccessed::READ));
  const double sram_rd_bw_utilization =
      CapUtilization(tsl::profiler::SafeDivide(
          sram_rd_gibibytes_per_second,
          get_peak_mem_bw(MemBwType::MEM_BW_TYPE_SRAM_RD)));
  metrics->add_bandwidth_utils(sram_rd_bw_utilization);
  double sram_rd_bytes =
      tsl::profiler::GibiToGiga(sram_rd_gibibytes_per_second) *
      tsl::profiler::PicoToNano(op_metrics.time_ps());

  const double sram_wr_gibibytes_per_second = tsl::profiler::GigaToGibi(
      GigaBytesPerSecondPerCore(op_metrics, MemorySpace::MEMORY_SPACE_ON_CHIP,
                                OpMetrics::MemoryAccessed::WRITE));
  const double sram_wr_bw_utilization =
      CapUtilization(tsl::profiler::SafeDivide(
          sram_wr_gibibytes_per_second,
          get_peak_mem_bw(MemBwType::MEM_BW_TYPE_SRAM_WR)));
  metrics->add_bandwidth_utils(sram_wr_bw_utilization);
  double sram_wr_bytes =
      tsl::profiler::GibiToGiga(sram_wr_gibibytes_per_second) *
      tsl::profiler::PicoToNano(op_metrics.time_ps());

  metrics->add_raw_bytes_accessed_array(hbm_bytes);
  metrics->add_raw_bytes_accessed_array(sram_rd_bytes);
  metrics->add_raw_bytes_accessed_array(sram_wr_bytes);
}

// Recursively insert "fused instruction" nodes (with raw flops).
void InsertFusedInstructions(const OpMetrics& op_metrics, Node* node,
                             OpProfileBuilder* builder) {
  if (!op_metrics.has_children()) return;
  for (const auto& child : op_metrics.children().metrics_db()) {
    Node* new_node = node->add_children();
    PopulateSymbolNode(child, new_node);
    new_node->mutable_metrics()->set_raw_flops(child.model_flops_v2());
    new_node->mutable_metrics()->set_bf16_flops(child.flops_v2());
    builder->AddChildMetrics(child, new_node);
    if (child.has_children()) {
      InsertFusedInstructions(child, new_node, builder);
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
    parent->set_flops_v2(child.flops_v2() + parent->flops_v2());
    parent->set_model_flops_v2(child.model_flops_v2() +
                               parent->model_flops_v2());
    parent->set_bytes_accessed(child.bytes_accessed() +
                               parent->bytes_accessed());
    parent->set_dma_stall_ps(child.dma_stall_ps() + parent->dma_stall_ps());
    CombineMemoryAccessedBreakdown(child.memory_accessed_breakdown(),
                                   parent->mutable_memory_accessed_breakdown());
  }
}

OpMetrics_MemoryAccessed_OperationType ConvertOperationType(
    FlatOpMetrics::MemoryAccessed::OperationType type) {
  return static_cast<OpMetrics_MemoryAccessed_OperationType>(type);
}

void CombineMemoryAccessed(
    const tsl::protobuf::RepeatedPtrField<FlatOpMetrics::MemoryAccessed>& src,
    tsl::protobuf::RepeatedPtrField<OpMetrics::MemoryAccessed>* dst) {
  for (const auto& src_mem : src) {
    bool found = false;
    for (auto& dst_mem : *dst) {
      if (dst_mem.memory_space() == src_mem.memory_space() &&
          dst_mem.operation_type() ==
              ConvertOperationType(src_mem.operation_type())) {
        dst_mem.set_bytes_accessed(dst_mem.bytes_accessed() +
                                   src_mem.bytes_accessed());
        found = true;
        break;
      }
    }
    if (!found) {
      auto* new_dst_mem = dst->Add();
      new_dst_mem->set_memory_space(src_mem.memory_space());
      new_dst_mem->set_bytes_accessed(src_mem.bytes_accessed());
      new_dst_mem->set_operation_type(
          ConvertOperationType(src_mem.operation_type()));
    }
  }
}

void CombineFlatOpMetricsToOpMetrics(const FlatOpMetrics& src, OpMetrics* dst,
                                     bool update_num_cores) {
  DCHECK(dst != nullptr);
  if (dst->occurrences() == 0) {
    dst->set_min_time_ps(src.min_time_ps());
  } else {
    dst->set_min_time_ps(std::min(src.min_time_ps(), dst->min_time_ps()));
  }
  dst->set_is_eager(dst->is_eager() || src.is_eager());
  dst->set_occurrences(src.occurrences() + dst->occurrences());
  dst->set_time_ps(src.time_ps() + dst->time_ps());
  dst->set_self_time_ps(src.self_time_ps() + dst->self_time_ps());
  dst->set_normalized_time_ps(src.normalized_time_ps() +
                              dst->normalized_time_ps());
  dst->set_flops(src.flops() + dst->flops());
  dst->set_model_flops(src.model_flops() + dst->model_flops());
  dst->set_flops_v2(src.flops_v2() + dst->flops_v2());
  dst->set_model_flops_v2(src.model_flops_v2() + dst->model_flops_v2());
  dst->set_bytes_accessed(src.bytes_accessed() + dst->bytes_accessed());
  dst->set_autotuned(dst->autotuned() || src.autotuned());
  if (update_num_cores) {
    dst->set_num_cores(src.num_cores() + dst->num_cores());
  }
  CombineMemoryAccessed(src.memory_accessed_breakdown(),
                        dst->mutable_memory_accessed_breakdown());

  dst->set_dma_stall_ps(src.dma_stall_ps() + dst->dma_stall_ps());
}

// Update child node metrics to parent node for FlatOpMetrics.
void UpdateNodeMetrics(const FlatOpMetrics& child, OpMetrics* parent) {
  DCHECK(parent != nullptr);
  parent->set_time_ps(child.self_time_ps() + parent->time_ps());
  parent->set_normalized_time_ps(child.normalized_time_ps() +
                                 parent->normalized_time_ps());
  parent->set_self_time_ps(child.self_time_ps() + parent->self_time_ps());

  if (ChildrenTimePs(child) == 0) {
    parent->set_flops(child.flops() + parent->flops());
    parent->set_model_flops(child.model_flops() + parent->model_flops());
    parent->set_flops_v2(child.flops_v2() + parent->flops_v2());
    parent->set_model_flops_v2(child.model_flops_v2() +
                               parent->model_flops_v2());
    parent->set_bytes_accessed(child.bytes_accessed() +
                               parent->bytes_accessed());
    parent->set_dma_stall_ps(child.dma_stall_ps() + parent->dma_stall_ps());
    CombineMemoryAccessed(child.memory_accessed_breakdown(),
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
  if (category == nullptr && deduplicated_node == nullptr) {
    PopulateSymbolNode(op_metrics, leaf);
    InsertFusedInstructions(op_metrics, leaf, this);
  } else {
    Metrics* m = leaf->mutable_metrics();
    m->set_raw_time(op_metrics.time_ps());
    m->set_raw_flops(op_metrics.model_flops_v2());
    leaf_op_metrics_[leaf] = &op_metrics;
  }
  return leaf;
}

Node* OpProfileBuilder::AddOpNode(const FlatOpMetrics& op_metrics,
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
  id_to_node_map_[op_metrics.op_id()] = leaf;
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

Node* OpProfileBuilder::LookupOrAddDeduplicatedNode(
    const FlatOpMetrics& op_metrics, Category* category) {
  std::string deduplicated_name = op_metrics.deduplicated_name().empty()
                                      ? op_metrics.hlo_name()
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

OpProfileBuilder::Category* OpProfileBuilder::LookupOrAddCategoryNode(
    const FlatOpMetrics& op_metrics, Program* program,
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
    uint64_t program_id) {
  Program* program = &programs_map_[program_id];
  if (program->node == nullptr) {
    program->node = root_->add_children();
    program->node->set_name(GenerateProgramName(program_id));
    if (program_serving_metadata_ != nullptr) {
      auto it = program_serving_metadata_->find(program_id);
      if (it != program_serving_metadata_->end() &&
          !it->second.model_id().empty() && it->second.batch_size() > 0 &&
          it->second.num_batches() > 0) {
        *program->node->mutable_program_serving_metadata() = it->second;
      }
    }
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

Node* OpProfileBuilder::GetOrAddProvenanceParentNode(
    const FlatOpMetrics& op_metrics, Program* program) {
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
    program = LookupOrAddProgramNode(op_metrics.hlo_module_id());
    UpdateNodeMetrics(op_metrics, &metrics_[program->node]);
  }

  // 2. Deal with nested grouping nodes (horizontal grouping), only accumulate
  // non-child ops Exclude ops with children ops to avoid double counting of
  // flops, bytes and time from children ops.
  bool has_children = ChildrenTimePs(op_metrics) > 0;
  bool has_sc_children = false;
  if (has_children) {
    for (const auto& child : op_metrics.children().metrics_db()) {
      if (child.core_type() == OpMetrics_TpuCoreType_SPARSE_CORE) {
        has_sc_children = true;
        break;
      }
    }
  }
  if (has_children && !has_sc_children) return;
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

    AddOpNode(op_metrics, category, deduplicated_node);
  }

  for (auto* node : nested_grouping_nodes) {
    // Per program combiner does not need to update OpMetrics.num_cores
    CombineOpMetrics(op_metrics, &metrics_[node], /*update_num_cores=*/false);
  }
}

void OpProfileBuilder::AddOp(const FlatOpMetrics& op_metrics) {
  // 1. Deal with nested parent nodes (vertical grouping)
  // op_metrics.time_ps in root node will be reset to total_time_ps later
  UpdateNodeMetrics(op_metrics, &metrics_[root_]);
  Program* program = nullptr;
  if (!IsIdleOp(op_metrics) && RequiresProgramGrouping(options_)) {
    program = LookupOrAddProgramNode(op_metrics.hlo_module_id());
    UpdateNodeMetrics(op_metrics, &metrics_[program->node]);
  }
  bool has_children = ChildrenTimePs(op_metrics) > 0;
  if (has_children) return;
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
    CombineFlatOpMetricsToOpMetrics(op_metrics, &metrics_[node],
                                    /*update_num_cores=*/false);
  }
}

void OpProfileBuilder::AddFusionAndSparseCoreOp(
    const FlatOpMetrics& op_metrics) {
  if ((op_metrics.core_type() == FlatOpMetrics::SPARSE_CORE) ||
      op_metrics.is_fusion_child()) {
    auto parent_id = op_metrics.parent_op_id();
    Node* leaf_node = root_;
    if (id_to_node_map_.contains(parent_id)) {
      leaf_node = id_to_node_map_[parent_id];
    } else {
      LOG(ERROR) << "[Skipping] Parent op id not found:"
                 << op_metrics.hlo_name();
      return;
    }
    Node* new_node = leaf_node->add_children();
    id_to_node_map_[op_metrics.op_id()] = new_node;  // TOPO SORTED ORDER
    PopulateSymbolNode(op_metrics, new_node);
    new_node->mutable_metrics()->set_raw_flops(op_metrics.model_flops_v2());
    new_node->mutable_metrics()->set_bf16_flops(op_metrics.flops_v2());
    // 1. Combine metrics into the new SparseCore node.
    CombineFlatOpMetricsToOpMetrics(op_metrics, &metrics_[new_node],
                                    /*update_num_cores=*/false);
  }
}

void OpProfileBuilder::AddChildMetrics(const OpMetrics& child,
                                       op_profile::Node* child_node) {
  CombineOpMetrics(child, &metrics_[child_node], /*update_num_cores=*/false);
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
  // Defer full metric computation (FLOPS, repeated bandwidth utils arrays,
  // etc.) until after children are sorted and pruned, which destroys >90% of
  // nodes. SortAndPruneChildren only requires raw_time and raw_flops.
  for (const auto& [node, op_metrics] : metrics_) {
    Metrics* m = node->mutable_metrics();
    m->set_raw_time(op_metrics.time_ps());
    m->set_raw_flops(op_metrics.model_flops_v2());
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
  SortAndPruneChildren(options_.children_per_node, level, root_,
                       &leaf_op_metrics_);

  auto populate_surviving_nodes = [&](auto& self, Node* node) -> void {
    if (auto leaf_it = leaf_op_metrics_.find(node);
        leaf_it != leaf_op_metrics_.end()) {
      PopulateOpMetricsNode(
          *leaf_it->second, peak_gigaflops_per_second_per_core,
          peak_mem_gibibytes_per_second_per_core, total_time_ps, node);
    } else if (auto it = metrics_.find(node); it != metrics_.end()) {
      PopulateOpMetricsNode(it->second, peak_gigaflops_per_second_per_core,
                            peak_mem_gibibytes_per_second_per_core,
                            total_time_ps, node);
    }
    for (Node& child : *node->mutable_children()) {
      self(self, &child);
    }
  };
  populate_surviving_nodes(populate_surviving_nodes, root_);

  if (options_.group_by_deduplicated_name) {
    FinalizeDeduplicatedNodes(root_);
  }
}

OpProfileBuilder::OpProfileBuilder(
    const OpProfileOptions& options,
    tensorflow::profiler::op_profile::Node* root,
    const tsl::protobuf::Map<uint64_t, std::string>* program_name_map,
    const ProgramServingMetadataMap* absl_nullable program_serving_metadata)
    : options_(options),
      root_(root),
      program_name_map_(program_name_map),
      program_serving_metadata_(program_serving_metadata) {
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

namespace {

void InternStackFramesInNode(
    Node* node, absl::flat_hash_map<std::string, uint32_t>& index_by_frame,
    tsl::protobuf::RepeatedPtrField<std::string>& table) {
  if (node->has_xla() && node->xla().has_source_info() &&
      !node->xla().source_info().stack_frame().empty()) {
    Node::XLAInstruction& xla = *node->mutable_xla();
    std::string* frame = xla.mutable_source_info()->mutable_stack_frame();
    auto [it, inserted] =
        index_by_frame.try_emplace(*frame, static_cast<uint32_t>(table.size()));
    if (inserted) *table.Add() = std::move(*frame);
    xla.set_stack_frame_index(it->second);
    xla.mutable_source_info()->clear_stack_frame();
  } else if (node->has_xla() && node->xla().has_source_info()) {
    // Drop empty stack frames so they are not serialized as "".
    node->mutable_xla()->mutable_source_info()->clear_stack_frame();
  }
  for (Node& child : *node->mutable_children()) {
    InternStackFramesInNode(&child, index_by_frame, table);
  }
}

}  // namespace

void InternStackFrames(op_profile::Profile* profile) {
  absl::flat_hash_map<std::string, uint32_t> index_by_frame;
  tsl::protobuf::RepeatedPtrField<std::string>& table =
      *profile->mutable_stack_frame_table();
  index_by_frame.reserve(table.size());
  for (int i = 0; i < table.size(); ++i) {
    index_by_frame.try_emplace(table[i], static_cast<uint32_t>(i));
  }
  // Only visit trees that are present; mutable_*() would otherwise create
  // empty trees and change the response.
  auto intern = [&](bool present,
                    Node* (op_profile::Profile::*mutable_root)()) {
    if (present) {
      InternStackFramesInNode((profile->*mutable_root)(), index_by_frame,
                              table);
    }
  };
  intern(profile->has_by_category(), &op_profile::Profile::mutable_by_category);
  intern(profile->has_by_program(), &op_profile::Profile::mutable_by_program);
  intern(profile->has_by_category_exclude_idle(),
         &op_profile::Profile::mutable_by_category_exclude_idle);
  intern(profile->has_by_program_exclude_idle(),
         &op_profile::Profile::mutable_by_program_exclude_idle);
  intern(profile->has_by_provenance(),
         &op_profile::Profile::mutable_by_provenance);
  intern(profile->has_by_provenance_exclude_idle(),
         &op_profile::Profile::mutable_by_provenance_exclude_idle);

  for (op_profile::NodePatch& patch : *profile->mutable_exclude_idle_patch()) {
    if (patch.has_replacement()) {
      InternStackFramesInNode(patch.mutable_replacement(), index_by_frame,
                              table);
    }
    if (patch.has_xla() && patch.xla().has_source_info()) {
      Node::XLAInstruction& xla = *patch.mutable_xla();
      if (!xla.source_info().stack_frame().empty()) {
        std::string* frame = xla.mutable_source_info()->mutable_stack_frame();
        auto [it, inserted] = index_by_frame.try_emplace(
            *frame, static_cast<uint32_t>(table.size()));
        if (inserted) *table.Add() = std::move(*frame);
        xla.set_stack_frame_index(it->second);
      }
      xla.mutable_source_info()->clear_stack_frame();
    }
  }
}

namespace {

using op_profile::NodePatch;

bool SameMetrics(const op_profile::Metrics& a, const op_profile::Metrics& b) {
  if (absl::bit_cast<uint64_t>(a.flops()) !=
          absl::bit_cast<uint64_t>(b.flops()) ||
      absl::bit_cast<uint64_t>(a.uncapped_flops()) !=
          absl::bit_cast<uint64_t>(b.uncapped_flops()) ||
      absl::bit_cast<uint64_t>(a.raw_time()) !=
          absl::bit_cast<uint64_t>(b.raw_time()) ||
      absl::bit_cast<uint64_t>(a.raw_flops()) !=
          absl::bit_cast<uint64_t>(b.raw_flops()) ||
      absl::bit_cast<uint64_t>(a.bf16_flops()) !=
          absl::bit_cast<uint64_t>(b.bf16_flops()) ||
      absl::bit_cast<uint64_t>(a.normalized_time_ps()) !=
          absl::bit_cast<uint64_t>(b.normalized_time_ps()) ||
      a.occurrences() != b.occurrences() ||
      absl::bit_cast<uint64_t>(a.avg_time_ps()) !=
          absl::bit_cast<uint64_t>(b.avg_time_ps())) {
    return false;
  }
  if (a.bandwidth_utils_size() != b.bandwidth_utils_size()) return false;
  for (int i = 0; i < a.bandwidth_utils_size(); ++i) {
    if (absl::bit_cast<uint64_t>(a.bandwidth_utils(i)) !=
        absl::bit_cast<uint64_t>(b.bandwidth_utils(i))) {
      return false;
    }
  }
  if (a.raw_bytes_accessed_array_size() != b.raw_bytes_accessed_array_size()) {
    return false;
  }
  for (int i = 0; i < a.raw_bytes_accessed_array_size(); ++i) {
    if (absl::bit_cast<uint64_t>(a.raw_bytes_accessed_array(i)) !=
        absl::bit_cast<uint64_t>(b.raw_bytes_accessed_array(i))) {
      return false;
    }
  }
  return true;
}

bool SameXlaInstruction(const Node::XLAInstruction& a,
                        const Node::XLAInstruction& b) {
  if (a.program_id() != b.program_id() || a.fingerprint() != b.fingerprint() ||
      a.computation_primitive_size() != b.computation_primitive_size() ||
      a.has_layout() != b.has_layout() ||
      a.has_source_info() != b.has_source_info() ||
      a.has_xprof_kernel_metadata() != b.has_xprof_kernel_metadata() ||
      a.has_stack_frame_index() != b.has_stack_frame_index() ||
      a.op() != b.op() || a.category() != b.category() ||
      a.provenance() != b.provenance() || a.expression() != b.expression()) {
    return false;
  }
  if (a.has_xprof_kernel_metadata() &&
      a.xprof_kernel_metadata() != b.xprof_kernel_metadata()) {
    return false;
  }
  if (a.has_stack_frame_index() &&
      a.stack_frame_index() != b.stack_frame_index()) {
    return false;
  }
  if (a.has_layout()) {
    const auto& la = a.layout().dimensions();
    const auto& lb = b.layout().dimensions();
    if (la.size() != lb.size()) return false;
    for (int i = 0; i < la.size(); ++i) {
      if (la[i].size() != lb[i].size() ||
          la[i].alignment() != lb[i].alignment() ||
          la[i].semantics() != lb[i].semantics()) {
        return false;
      }
    }
  }
  if (a.has_source_info()) {
    const auto& sa = a.source_info();
    const auto& sb = b.source_info();
    if (sa.line_number() != sb.line_number() ||
        sa.has_stack_frame() != sb.has_stack_frame() ||
        sa.file_name() != sb.file_name()) {
      return false;
    }
    if (sa.has_stack_frame() && sa.stack_frame() != sb.stack_frame()) {
      return false;
    }
  }
  return true;
}

bool SameProgramServingMetadata(const Node& a, const Node& b) {
  if (a.has_program_serving_metadata() != b.has_program_serving_metadata()) {
    return false;
  }
  if (!a.has_program_serving_metadata()) return true;
  const auto& ma = a.program_serving_metadata();
  const auto& mb = b.program_serving_metadata();
  return ma.model_id() == mb.model_id() && ma.batch_size() == mb.batch_size() &&
         ma.num_batches() == mb.num_batches();
}

// Appends patches describing how `derived` differs from `base`.
void DiffNode(const Node& base, const Node& derived,
              std::vector<uint32_t>& path,
              tsl::protobuf::RepeatedPtrField<NodePatch>* patches) {
  auto add_patch = [&]() {
    NodePatch* patch = patches->Add();
    patch->mutable_path()->Assign(path.begin(), path.end());
    return patch;
  };
  // Match derived children to base children by name, in order.
  std::vector<std::pair<int, int>> matched;  // (base index, derived index)
  std::vector<uint32_t> removed;
  bool aligned = true;
  int b = 0;
  for (int d = 0; d < derived.children_size(); ++d) {
    while (b < base.children_size() &&
           base.children(b).name() != derived.children(d).name()) {
      removed.push_back(b++);
    }
    if (b == base.children_size()) {
      aligned = false;
      break;
    }
    matched.emplace_back(b++, d);
  }
  while (aligned && b < base.children_size()) removed.push_back(b++);
  if (!aligned || base.contents_case() != derived.contents_case() ||
      base.has_metrics() != derived.has_metrics() ||
      !SameProgramServingMetadata(base, derived)) {
    *add_patch()->mutable_replacement() = derived;
    return;
  }
  NodePatch patch;
  bool changed = !removed.empty();
  if (base.name() != derived.name()) {
    patch.set_name(derived.name());
    changed = true;
  }
  if (derived.has_metrics() &&
      !SameMetrics(base.metrics(), derived.metrics())) {
    *patch.mutable_metrics() = derived.metrics();
    changed = true;
  }
  if (derived.has_xla() && !SameXlaInstruction(base.xla(), derived.xla())) {
    *patch.mutable_xla() = derived.xla();
    changed = true;
  }
  if (base.num_children() != derived.num_children()) {
    patch.set_num_children(derived.num_children());
    changed = true;
  }
  if (changed) {
    patch.mutable_removed_children()->Assign(removed.begin(), removed.end());
    patch.mutable_path()->Assign(path.begin(), path.end());
    *patches->Add() = std::move(patch);
  }
  for (const auto& [base_index, derived_index] : matched) {
    path.push_back(base_index);
    DiffNode(base.children(base_index), derived.children(derived_index), path,
             patches);
    path.pop_back();
  }
}

}  // namespace

void ReplaceExcludeIdleTreesWithPatches(op_profile::Profile* profile) {
  struct TreePair {
    const Node* base;
    Node* derived;
    void (op_profile::Profile::*clear_derived)();
  };
  std::vector<TreePair> pairs;
  if (profile->has_by_category() && profile->has_by_category_exclude_idle()) {
    pairs.push_back({&profile->by_category(),
                     profile->mutable_by_category_exclude_idle(),
                     &op_profile::Profile::clear_by_category_exclude_idle});
  }
  if (profile->has_by_program() && profile->has_by_program_exclude_idle()) {
    pairs.push_back({&profile->by_program(),
                     profile->mutable_by_program_exclude_idle(),
                     &op_profile::Profile::clear_by_program_exclude_idle});
  }
  if (profile->has_by_provenance() &&
      profile->has_by_provenance_exclude_idle()) {
    pairs.push_back({&profile->by_provenance(),
                     profile->mutable_by_provenance_exclude_idle(),
                     &op_profile::Profile::clear_by_provenance_exclude_idle});
  }
  // Patches are only unambiguous when a single tree pair is present.
  if (pairs.size() != 1) return;
  profile->clear_exclude_idle_patch();
  std::vector<uint32_t> path;
  DiffNode(*pairs[0].base, *pairs[0].derived, path,
           profile->mutable_exclude_idle_patch());
  if (profile->exclude_idle_patch().empty()) {
    // Emit an identity root patch so clients know `derived` was cleared and
    // rebuild it as a clone of `base` when `base` and `derived` are identical.
    profile->add_exclude_idle_patch();
  }
  (profile->*pairs[0].clear_derived)();
}

}  // namespace profiler
}  // namespace tensorflow
