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

#include "xprof/utils/hlo_module_map.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_utils.h"
#include "xprof/utils/hlo_proto_map.h"
#include "xprof/utils/hlo_proto_to_module.h"
#include "xprof/utils/lazy.h"
#include "xprof/utils/performance_info_wrapper.h"

namespace tensorflow {
namespace profiler {

HloInstructionWrapper::HloInstructionWrapper(
    const xla::HloInstruction* instr,
    const HloCostAnalysisWrapper* cost_analysis)
    : instr_(instr),
      fingerprint_(xprof::ThreadSafeLazy<uint64_t>([instr] {
        return tsl::Fingerprint64(
            instr->ToString(xla::HloPrintOptions::Fingerprint()));
      })),
      op_full_name_(
          tsl::profiler::TraceMeOp(Metadata().op_name(), Metadata().op_type())),
      tf_op_name_(tsl::profiler::TfOpFullname(Metadata().op_type(),
                                              Metadata().op_name())),
      category_(instr_->ToCategory()),
      expression_(UncachedExpression(*instr_, false, kMaxHlolNameSize)),
      deduplicated_name_(instr_->metadata().deduplicated_name()) {
  if (cost_analysis != nullptr) {
    performance_info_wrapper_ =
        tensorflow::profiler::PerformanceInfoWrapper::Create(cost_analysis,
                                                             instr);
    ProcessXlaCostAnalysis(cost_analysis->GetXlaCostAnalysis());
  }
}

std::vector<std::string> HloInstructionWrapper::InputTensors() const {
  std::vector<std::string> input_tensors;
  input_tensors.reserve(instr_->operand_count());
  for (const xla::HloInstruction* operand : instr_->operands()) {
    input_tensors.push_back(operand->shape().ToString());
  }
  return input_tensors;
}

std::vector<std::string> HloInstructionWrapper::OutputTensors() const {
  return {instr_->shape().ToString()};
}

HloModuleWrapper::HloModuleWrapper(
    const xla::HloProto& hlo_proto,
    std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper)
    : HloModuleWrapper(ConvertHloProtoToModuleIgnoringErrors(hlo_proto),
                       std::move(cost_analysis_wrapper)) {}

HloModuleWrapper::HloModuleWrapper(
    std::unique_ptr<xla::HloModule> module,
    std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper)
    : module_(std::move(module)) {
  if (module_ == nullptr) return;

  if (cost_analysis_wrapper != nullptr) {
    absl::Status status = tensorflow::profiler::InitializeHloCostAnalysis(
        *module_, *cost_analysis_wrapper->GetXlaCostAnalysis());
    if (!status.ok()) {
      LOG(ERROR) << "Failed to initialize xla::HloCostAnalysis for module: "
                 << module_->name() << " with status: " << status;
      cost_analysis_wrapper.reset();
    }
  }

  // Populate instructions_by_name_ with module.
  std::vector<const xla::HloInstruction*> all_instrs;
  all_instrs.reserve(module_->instruction_count());
  for (const xla::HloComputation* computation : module_->computations()) {
    for (const xla::HloInstruction* instr : computation->instructions()) {
      all_instrs.push_back(instr);
    }
  }
  const HloCostAnalysisWrapper* raw_cost_analysis = cost_analysis_wrapper.get();
  instructions_by_name_.reserve(all_instrs.size());
  if (all_instrs.size() >= 1024) {
    std::vector<std::optional<HloInstructionWrapper>> temp_wrappers(
        all_instrs.size());
    constexpr size_t kNumChunks = 16;
    const size_t chunk_size = (all_instrs.size() + kNumChunks - 1) / kNumChunks;
    {
      tsl::thread::ThreadPool pool(tsl::Env::Default(), "HloInstrWrap",
                                   kNumChunks);
      for (size_t c = 0; c < kNumChunks; ++c) {
        const size_t start = c * chunk_size;
        if (start >= all_instrs.size()) break;
        const size_t end = std::min(start + chunk_size, all_instrs.size());
        pool.Schedule([&, start, end]() {
          for (size_t i = start; i < end; ++i) {
            temp_wrappers[i].emplace(all_instrs[i], raw_cost_analysis);
          }
        });
      }
    }
    for (size_t i = 0; i < all_instrs.size(); ++i) {
      instructions_by_name_.try_emplace(all_instrs[i]->name(),
                                        std::move(*temp_wrappers[i]));
    }
  } else {
    for (const xla::HloInstruction* instr : all_instrs) {
      instructions_by_name_.try_emplace(
          instr->name(), HloInstructionWrapper(instr, raw_cost_analysis));
    }
  }
  // Gather nested fusion instructions.
  for (const xla::HloComputation* computation : module_->computations()) {
    // Some modules still seem to have "dead" fusions computations. In this
    // case, IsFusionComputation() = true but there is no parent
    // FusionInstruction().
    if (computation->FusionInstruction() != nullptr) {
      GatherFusionInstructions(computation->FusionInstruction());
    }
  }
}

// Function to gather all the instructions in a fusion computation.
void HloModuleWrapper::GatherFusionInstructions(xla::HloInstruction* inst) {
  HloInstructionWrapper* fused_inst_wrapper =
      GetMutableHloInstruction(inst->name());
  DCHECK(fused_inst_wrapper != nullptr);
  if (!fused_inst_wrapper->FusedChildren().empty()) return;
  for (auto* fused : inst->fused_instructions()) {
    const auto child_inst_wrapper = GetHloInstruction(fused->name());
    DCHECK(child_inst_wrapper != nullptr);
    fused_inst_wrapper->AddFusedChild(child_inst_wrapper);
    if (fused->opcode() == xla::HloOpcode::kFusion) {
      GatherFusionInstructions(fused);
    }
  }
}

HloInstructionWrapper* HloModuleWrapper::GetMutableHloInstruction(
    absl::string_view hlo_name) {
  auto it = instructions_by_name_.find(hlo_name);
  if (it != instructions_by_name_.end()) return &it->second;
  return nullptr;
}

const HloInstructionWrapper* HloModuleWrapper::GetHloInstruction(
    absl::string_view hlo_name) const {
  auto it = instructions_by_name_.find(hlo_name);
  if (it != instructions_by_name_.end()) return &it->second;
  return nullptr;
}

std::string HloInstructionWrapper::source_info() const {
  if (!Metadata().source_file().empty()) {
    return absl::StrCat(tsl::io::Basename(Metadata().source_file()), ":",
                        Metadata().source_line());
  } else {
    return std::string();
  }
}

void AddHloProto(HloModuleMap& hlo_module_map, uint64_t program_id,
                 const xla::HloProto& hlo_proto,
                 std::unique_ptr<HloCostAnalysisWrapper> cost_analysis) {
  auto hlo_module = ConvertHloProtoToModule(hlo_proto);
  if (!hlo_module.ok()) {
    LOG(ERROR) << hlo_module.status();
    return;
  }
  hlo_module_map.try_emplace(program_id,
                             HloModuleWrapper(std::move(hlo_module).value(),
                                              std::move(cost_analysis)));
}

void ProcessHloModuleMapFromXSpace(
    HloModuleMap& hlo_module_map, const XSpace* space,
    tensorflow::profiler::HloCostAnalysisWrapper::Factory&
        create_cost_analysis,
    tsl::protobuf::Map<uint64_t, std::string>* program_id_to_name_map) {
  auto hlo_protos = ParseHloProtosFromXSpace(*space);
  if (program_id_to_name_map != nullptr) {
    for (const auto& [program_id, hlo_proto] : hlo_protos) {
      (*program_id_to_name_map)[program_id] = hlo_proto->hlo_module().name();
    }
  }
  if (hlo_protos.size() <= 1) {
    for (auto& [program_id, hlo_proto] : hlo_protos) {
      AddHloProto(hlo_module_map, program_id, *hlo_proto,
                  create_cost_analysis());
    }
    return;
  }
  std::vector<std::pair<uint64_t, std::unique_ptr<const xla::HloProto>>>
      entries;
  entries.reserve(hlo_protos.size());
  for (auto& [program_id, hlo_proto] : hlo_protos) {
    entries.emplace_back(program_id, std::move(hlo_proto));
  }
  std::vector<std::unique_ptr<HloCostAnalysisWrapper>> cost_analyses;
  cost_analyses.reserve(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    cost_analyses.push_back(create_cost_analysis());
  }
  std::vector<std::optional<HloModuleWrapper>> wrappers(entries.size());
  {
    const int num_threads = std::min<int>(entries.size(), 16);
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "HloModuleMap",
                                 num_threads);
    for (size_t i = 0; i < entries.size(); ++i) {
      pool.Schedule([&, i]() {
        auto hlo_module = ConvertHloProtoToModule(*entries[i].second);
        if (!hlo_module.ok()) {
          LOG(ERROR) << hlo_module.status();
          return;
        }
        wrappers[i].emplace(std::move(hlo_module).value(),
                            std::move(cost_analyses[i]));
      });
    }
  }
  hlo_module_map.reserve(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    if (wrappers[i].has_value()) {
      hlo_module_map.try_emplace(entries[i].first, std::move(*wrappers[i]));
    }
  }
}

absl::Status InitializeHloCostAnalysis(const xla::HloModule& hlo_module,
                                       xla::HloCostAnalysis& cost_analysis) {
  const xla::HloComputation* hlo_computation = hlo_module.entry_computation();
  cost_analysis.ReserveVisitStates(hlo_computation->instruction_count());
  absl::Status analysis_status = hlo_computation->Accept(&cost_analysis);
  if (analysis_status.ok()) {
    cost_analysis.DestroyVisitState();
  }
  return analysis_status;
}

}  // namespace profiler
}  // namespace tensorflow
