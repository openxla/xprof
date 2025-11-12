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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <initializer_list>
#include <deque>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "google/protobuf/json/json.h"
#include "third_party/stablehlo/stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Interfaces/DataLayoutInterfaces.h"
#include "xprof/utils/hlo_module_utils.h"
#include "xprof/utils/hlo_proto_map.h"
#include "xprof/utils/hlo_proto_to_module.h"
#include "xprof/utils/performance_info_wrapper.h"

#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Operation.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"

#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Value.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "xprof/utils/backend_configs.pb.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/PassManager.h"
#include "third_party/py/jax/jaxlib/mosaic/dialect/tpu/transforms/serde.h"
#include "third_party/py/jax/jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Math/IR/Math.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "google/protobuf/util/json_util.h"

namespace CustomCallCostEstimator {

// Represents the computational cost of an operation.
struct OperationCost {
  uint64_t flops = 0;
  uint64_t bytes_consumed = 0;
};

// Base class for operation cost estimators.
class OperationCostEstimator {
 public:
    virtual ~OperationCostEstimator() = default;
    virtual OperationCost Estimate(mlir::Operation* op) const = 0;
};

  // Estimator for element-wise operations.
class ElementWiseOpEstimator : public OperationCostEstimator {
 public:
    OperationCost Estimate(mlir::Operation* op) const override {
      OperationCost cost;
      if (op->getNumResults() > 0) {
        mlir::Value result = op->getResult(0);
        mlir::Type result_type = result.getType();
        const auto& data_layout = mlir::DataLayout::closest(op);
        cost.bytes_consumed = data_layout.getTypeSize(result_type);
        if (auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(result_type)) {
          if (shaped_type.hasStaticShape()) {
            cost.flops = shaped_type.getNumElements();
          }
          // For dynamic shapes, we cannot calculate flops or memory statically.
        } else {
          cost.flops = 1;  // Scalar type
        }
      }
      return cost;
    }
};

  // Estimator for memory-only operations like constant or load.
class MemoryOnlyOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    if (op->getNumResults() > 0) {
      mlir::Value result = op->getResult(0);
      const auto& data_layout = mlir::DataLayout::closest(op);
      cost.bytes_consumed = data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for matrix multiplication operations.
class MatmulOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    if (op->getNumOperands() >= 2 && op->getNumResults() == 1) {
      auto lhs = op->getOperand(0);
      auto result = op->getResult(0);
      auto lhs_type = mlir::dyn_cast<mlir::ShapedType>(lhs.getType());
      auto result_type = mlir::dyn_cast<mlir::ShapedType>(result.getType());

      if (lhs_type && result_type && lhs_type.hasStaticShape() &&
          result_type.hasStaticShape() && lhs_type.getRank() == 2 &&
          result_type.getRank() == 2) {
        auto result_shape = result_type.getShape();
        int64_t m = result_shape[0];
        int64_t n = result_shape[1];
        auto lhs_shape = lhs_type.getShape();
        // Infer K from the non-M dimension of LHS. This is a simplification
        // that works for non-transposed inputs.
        int64_t k = (lhs_shape[0] == m) ? lhs_shape[1] : lhs_shape[0];
        cost.flops = 2 * m * n * k;
        const auto& data_layout = mlir::DataLayout::closest(op);
        cost.bytes_consumed = data_layout.getTypeSize(result.getType());
      }
    }
    return cost;
  }
};

// Estimator for multi-reduction operations.
class MultiReductionOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    if (op->getNumOperands() > 0) {
      mlir::Value input = op->getOperand(0);
      if (auto shaped_type =
              mlir::dyn_cast<mlir::ShapedType>(input.getType())) {
        if (shaped_type.hasStaticShape()) {
          cost.flops = shaped_type.getNumElements();
        }
      }
    }
    if (op->getNumResults() > 0) {
      mlir::Value result = op->getResult(0);
      const auto& data_layout = mlir::DataLayout::closest(op);
      cost.bytes_consumed = data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// A singleton class to manage and dispatch cost estimations for MLIR ops.
class CostModel {
 public:
  static const CostModel& GetInstance() {
    static absl::NoDestructor<CostModel> instance;
    return *instance;
  }

  OperationCost GetOperationCost(mlir::Operation* op) const {
    auto it = estimators_.find(op->getName().getStringRef());
    if (it != estimators_.end()) {
      return it->second->Estimate(op);
    }
    // Return zero cost for unknown or no-cost ops like control flow, yield etc.
    return {};
  }

 private:
  friend class absl::NoDestructor<CostModel>;
  CostModel() {
    // Register estimators for different operation kinds.
    // NOLINTBEGIN
    RegisterEstimator<ElementWiseOpEstimator>(
        {"arith.cmpi", "arith.extui",
        "vector.broadcast", "arith.muli",
        "arith.index_cast", "arith.maximumf",
        "arith.subf", "math.exp",
        "arith.addf", "arith.divf",
        "arith.cmpf", "arith.select",
        "arith.mulf", "arith.truncf"});
    // NOLINTEND
    RegisterEstimator<MemoryOnlyOpEstimator>(
        {"arith.constant", "vector.load"});

    RegisterEstimator<MatmulOpEstimator>({"tpu.matmul"});

    RegisterEstimator<MultiReductionOpEstimator>(
        {"vector.multi_reduction"});
  }

  template <typename T>
  void RegisterEstimator(std::initializer_list<absl::string_view> op_names) {
    auto estimator = std::make_unique<T>();
    for (const auto& op_name : op_names) {
      estimators_[op_name] = estimator.get();
    }
    owned_estimators_.push_back(std::move(estimator));
  }

  absl::flat_hash_map<absl::string_view, const OperationCostEstimator*>
      estimators_;
  std::vector<std::unique_ptr<OperationCostEstimator>> owned_estimators_;
};

void calculateOperationCost(mlir::Operation* op,
                            uint64_t& block_bytes_consumed,
                            uint64_t& block_flops) {
  const auto& cost_model = CostModel::GetInstance();
  OperationCost cost = cost_model.GetOperationCost(op);
  block_bytes_consumed += cost.bytes_consumed;
  block_flops += cost.flops;
}

void calculateCustomCallCost(const xla::HloInstruction& hlo_instruction,
  absl::flat_hash_map<std::string,
    std::pair<uint64_t, uint64_t>>& custom_call_block_costs) {
    if (!hlo_instruction.has_backend_config()) {
        LOG(INFO) << "Backend config not found For Custom Call "
          << hlo_instruction.name();
        return;
    }
  google::protobuf::json::ParseOptions options;
  options.ignore_unknown_fields = true;
  xprof::BackendConfig config;

  auto status = google::protobuf::util::JsonStringToMessage(
      hlo_instruction.raw_backend_config_string(), &config, options);
  if ( (!status.ok()) || (!config.has_custom_call_config()) ) {
        LOG(INFO) << "Custom call config not found "
        << hlo_instruction.name();
  }
  xprof::CustomCallConfig custom_call_config = config.custom_call_config();
  // OLD CODE
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(true);
  context.loadDialect<mlir::func::FuncDialect,
                      mlir::scf::SCFDialect,
                      mlir::vector::VectorDialect,
                      mlir::math::MathDialect,
                      mlir::stablehlo::StablehloDialect,
                      mlir::mhlo::MhloDialect,
                      mlir::arith::ArithDialect,
                      mlir::memref::MemRefDialect,
                      mlir::LLVM::LLVMDialect,
                      mlir::tpu::TPUDialect>();
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> mlir_op_ref
      = xla::ParseMlirModuleString(
        static_cast<std::string>(custom_call_config.body()),
        context);

  if (!mlir_op_ref.ok()) {
    LOG(INFO) << "Failed to parse MLIR module for custom call "
        << hlo_instruction.name() << " with status: "
        << mlir_op_ref.status();
    return;
  }
  bool verify = false;
  mlir::OwningOpRef<mlir::ModuleOp>& module_op = mlir_op_ref.value();
  auto manager =
      mlir::PassManager::on<mlir::ModuleOp>(module_op->getContext());
  manager.enableVerifier(verify);
  manager.addPass(mlir::tpu::createMosaicSerdePass(
    mlir::tpu::MosaicSerdePassOptions{.serialize = false}));
  if (mlir::failed(manager.run(module_op.get()))) {
    LOG(WARNING) << "Skipping MosaicSerdePass for custom call "
              << hlo_instruction.name();
  }
  mlir::Operation* module_operation = module_op->getOperation();
  std::deque<mlir::Region*> queue{&(module_operation->getRegion(0))};
  int64_t block_counter = 0;
  uint64_t block_bytes_consumed = 0;
  uint64_t block_flops = 0;
  while (!queue.empty()) {
    mlir::Region* region = queue.front();
    queue.pop_front();
    for (mlir::Block& block : *region) {
      if (block.empty()) {
        continue;
      }
      for (mlir::Operation& op : block) {
        calculateOperationCost(&op, block_bytes_consumed, block_flops);
      }
      auto block_name = absl::StrCat("__block_", block_counter++);
      custom_call_block_costs[block_name] =
      std::make_pair(block_flops, block_bytes_consumed);
      block_bytes_consumed = block_flops = 0;
      for (mlir::Operation& op : block.without_terminator()) {
        for (mlir::Region& region : op.getRegions()) {
          if (!region.empty()) {
            queue.push_back(&region);
          }
        }
      }
    }
  }
}

}  // namespace CustomCallCostEstimator

namespace tensorflow {
namespace profiler {

void HloInstructionWrapper::SetCustomCallBlockCosts(){
  CustomCallCostEstimator::calculateCustomCallCost
    (*instr_, custom_call_block_costs_);
}

HloInstructionWrapper::HloInstructionWrapper(
    const xla::HloInstruction* instr,
    const HloCostAnalysisWrapper* cost_analysis)
    : instr_(instr),
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
  if (category_ == "custom-call") {
    SetCustomCallBlockCosts();
  }
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
  for (const xla::HloComputation* computation : module_->computations()) {
    for (const xla::HloInstruction* instr : computation->instructions()) {
      instructions_by_name_.try_emplace(
          instr->name(),
          HloInstructionWrapper(instr, cost_analysis_wrapper.get()));
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
        create_cost_analysis) {
  for (auto& [program_id, hlo_proto] : ParseHloProtosFromXSpace(*space)) {
    AddHloProto(hlo_module_map, program_id, *hlo_proto, create_cost_analysis());
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
