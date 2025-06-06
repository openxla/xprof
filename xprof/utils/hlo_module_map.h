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

#ifndef XPROF_UTILS_HLO_MODULE_MAP_H_
#define XPROF_UTILS_HLO_MODULE_MAP_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_utils.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace tensorflow {
namespace profiler {

class HloInstructionInterface {
 public:
  virtual ~HloInstructionInterface() = default;
  virtual absl::string_view Name() const = 0;
  virtual xla::HloOpcode HloOpcode() const = 0;
  virtual absl::string_view Category() const = 0;
  virtual std::string HloOpcodeString() const = 0;
  virtual const xla::OpMetadata& Metadata() const = 0;
  virtual size_t flops() const = 0;
  virtual size_t bytes_accessed() const = 0;
  virtual std::string_view op_full_name() const = 0;
  virtual std::string_view TfOpName() const = 0;
  virtual std::string source_info() const = 0;
  virtual bool isRoot() const = 0;
  virtual bool IsFusion() const = 0;
  virtual const std::string& Expression() const = 0;
  virtual std::string_view DeduplicatedName() const = 0;

  virtual void ProcessXlaCostAnalysis(
      const xla::HloCostAnalysis* cost_analysis) = 0;
  virtual std::string OpLocationStack(int32_t frame_id) const = 0;
  virtual tsl::profiler::OpSourceInfo SourceInfo() const = 0;
  virtual const ::tensorflow::profiler::PerformanceInfoWrapper*
  GetPerformanceInfoWrapper() const = 0;
};

// This wrapper allows caching the results of HloInstruction methods.
// This wrapper is not thread safe.
class HloInstructionWrapper : public HloInstructionInterface {
 public:
  explicit HloInstructionWrapper(
      const xla::HloInstruction* instr,
      const tensorflow::profiler::HloCostAnalysisWrapper* cost_analysis =
          nullptr);

  // Non copyable
  HloInstructionWrapper(const HloInstructionWrapper&) = delete;
  HloInstructionWrapper& operator=(const HloInstructionWrapper&) = delete;
  // Movable.
  HloInstructionWrapper(HloInstructionWrapper&&) = default;
  HloInstructionWrapper& operator=(HloInstructionWrapper&&) = default;

  absl::string_view Name() const override { return instr_->name(); }

  xla::HloOpcode HloOpcode() const override { return instr_->opcode(); }

  absl::string_view Category() const override { return category_; }

  std::string HloOpcodeString() const override {
    return std::string(xla::HloOpcodeString(instr_->opcode()));
  }

  const xla::OpMetadata& Metadata() const override {
    return instr_->metadata();
  }

  size_t flops() const override { return flops_; }
  size_t bytes_accessed() const override { return bytes_accessed_; }

  std::string_view op_full_name() const override { return op_full_name_; }
  std::string_view TfOpName() const override { return tf_op_name_; }
  std::string source_info() const override;

  bool isRoot() const override { return instr_->IsRoot(); }
  bool IsFusion() const override { return !fused_children_.empty(); };
  std::string_view DeduplicatedName() const override {
    return deduplicated_name_;
  }

  void ProcessXlaCostAnalysis(
      const xla::HloCostAnalysis* cost_analysis) override {
    if (cost_analysis == nullptr) return;
    flops_ = cost_analysis->flop_count(*instr_);
    bytes_accessed_ = cost_analysis->bytes_accessed(*instr_);
  }

  const std::string& Expression() const override { return expression_; }

  void AddFusedChild(const HloInstructionWrapper* child) {
    fused_children_.push_back(child);
  };

  const std::vector<const HloInstructionWrapper*>& FusedChildren() const {
    return fused_children_;
  }

  std::string OpLocationStack(int32_t frame_id) const override {
    return GetOpLocationStack(frame_id, *instr_);
  }

  tsl::profiler::OpSourceInfo SourceInfo() const override {
    return GetSourceInfo(*instr_);
  }

  const ::tensorflow::profiler::PerformanceInfoWrapper*
  GetPerformanceInfoWrapper() const override {
    return performance_info_wrapper_.get();
  }

 private:
  const xla::HloInstruction* instr_;
  std::vector<const HloInstructionWrapper*> fused_children_;
  std::string op_full_name_;
  std::string tf_op_name_;
  size_t flops_ = 0;
  size_t bytes_accessed_ = 0;
  std::string category_;
  std::string expression_;
  std::string deduplicated_name_;
  std::unique_ptr<tensorflow::profiler::PerformanceInfoWrapper>
      performance_info_wrapper_;
};

// Helper class for accessing HloModule.
template <class T>
class HloModuleInterface {
 public:
  // If the module contains no instructions.
  bool Empty();
  absl::string_view Name();
  // Function to populated nested childs= instructions in a fusion.
  void GatherFusionInstructions(xla::HloInstruction* inst);

  auto Instructions() const {
    std::vector<const T*> result;
    for (auto& [name, instr] : instructions_by_name_) {
      result.push_back(&instr);
    }
    return result;
  }

 protected:
  // Map of HloInstructionWrappers by name.
  using HloInstructionMap = absl::flat_hash_map<absl::string_view, T>;
  HloInstructionMap instructions_by_name_;
};

// Wraps HLO module and provides an interface that maps HLO names to
// HloInstructionWrappers.
class HloModuleWrapper : public HloModuleInterface<HloInstructionWrapper> {
 public:
  explicit HloModuleWrapper(
      const xla::HloProto& hlo_proto,
      std::unique_ptr<HloCostAnalysisWrapper> cost_analysis = nullptr);

  explicit HloModuleWrapper(
      std::unique_ptr<xla::HloModule> module,
      std::unique_ptr<HloCostAnalysisWrapper> cost_analysis = nullptr);

  const HloInstructionWrapper* GetHloInstruction(
      absl::string_view hlo_name) const;
  HloInstructionWrapper* GetMutableHloInstruction(absl::string_view hlo_name);

  bool Empty() const { return instructions_by_name_.empty(); }

  absl::string_view Name() const { return module_->name(); }
  void GatherFusionInstructions(xla::HloInstruction* inst);

 private:
  std::unique_ptr<xla::HloModule> module_;
};

// Map of HloModuleWrappers by program_id.
using HloModuleMap =
    absl::flat_hash_map<uint64_t /*program_id*/, HloModuleWrapper>;

void AddHloProto(
    HloModuleMap& hlo_module_map, uint64_t program_id,
    const xla::HloProto& hlo_proto,
    std::unique_ptr<HloCostAnalysisWrapper> cost_analysis = nullptr);

// Process HloModuleMap from single XSpace.
void ProcessHloModuleMapFromXSpace(
    HloModuleMap& hlo_module_map, const XSpace* space,
    tensorflow::profiler::HloCostAnalysisWrapper::Factory&
        create_cost_analysis);

// WARNING: The returned pointer will be invalidated if HloModuleMap is mutated.
inline const HloModuleWrapper* GetHloModule(const HloModuleMap* hlo_module_map,
                                            uint64_t program_id) {
  if (hlo_module_map == nullptr) return nullptr;
  auto iter = hlo_module_map->find(program_id);
  if (iter == hlo_module_map->end()) return nullptr;
  return &iter->second;
}

inline const HloInstructionWrapper* GetHloInstruction(
    const HloModuleMap& hlo_module_map, std::optional<uint64_t> program_id,
    absl::string_view hlo_name) {
  if (!program_id.has_value()) return nullptr;
  const auto* hlo_module = GetHloModule(&hlo_module_map, *program_id);
  if (hlo_module == nullptr) return nullptr;
  return hlo_module->GetHloInstruction(hlo_name);
}

// Initialize HloCostAnalysis for the given HloModule.
absl::Status InitializeHloCostAnalysis(const xla::HloModule& hlo_module,
                                       xla::HloCostAnalysis& cost_analysis);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_HLO_MODULE_MAP_H_
