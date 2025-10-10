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

#include <cstddef>
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
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/AffineExpr.h"
#include "google/protobuf/json/json.h"
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
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OperationSupport.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"

#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Value.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/TypeID.h"
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

#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/Casting.h"


namespace CustomCallCostEstimator {

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
      for (mlir::Value result : op->getResults()) {
        mlir::Type result_type = result.getType();
        auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(result_type);
        if (!shaped_type) {
          cost.flops += 1;  // Scalar type
        } else if (shaped_type.hasStaticShape()) {
          cost.flops += shaped_type.getNumElements();
        }
      }
      const auto& data_layout = mlir::DataLayout::closest(op);
      for (mlir::Value operand : op->getOperands()) {
        cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
      }
      for (mlir::Value result : op->getResults()) {
        cost.bytes_consumed += data_layout.getTypeSize(result.getType());
      }
      return cost;
    }
};

  // Estimator for memory-only operations
class MemoryOnlyOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    if (op->getNumResults() == 0) {
      return cost;
    }
    mlir::Value result = op->getResult(0);
    const auto& data_layout = mlir::DataLayout::closest(op);
    cost.bytes_consumed = data_layout.getTypeSize(result.getType());
    return cost;
  }
};

// Estimator for Matrix Multiplication operations.
class MatmulOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;

    mlir::tpu::MatmulOp matmul_op = llvm::cast<mlir::tpu::MatmulOp>(op);

    auto lhs_st = llvm::cast<mlir::ShapedType>(matmul_op.getLhs().getType());
    auto rhs_st = llvm::cast<mlir::ShapedType>(matmul_op.getRhs().getType());
    auto acc_st = llvm::cast<mlir::ShapedType>(matmul_op.getAcc().getType());
    auto result_st = llvm::cast<mlir::ShapedType>(matmul_op.getResult().
        getType());

    if (!lhs_st || !lhs_st.hasStaticShape() || !rhs_st ||
        !rhs_st.hasStaticShape() || !result_st || !result_st.hasStaticShape()) {
      return cost;
    }

    uint64_t contracting_size_prod = 1;
    mlir::tpu::DotDimensionNumbersAttr dim_numbers =
        matmul_op.getDimensionNumbersAttr();

    if (dim_numbers) {
      auto contracting_dims = dim_numbers.getLhsContractingDims();
      for (int64_t d : contracting_dims) {
        if (d >= lhs_st.getRank()) {
          LOG(ERROR) << "Invalid contracting dimension: " << d
                     << " for lhs rank: " << lhs_st.getRank();
          return cost;
        }
        contracting_size_prod *= lhs_st.getShape()[d];
      }
    } else {
      // If dim_numbers is not present, infer contracting dim from transpose
      // attributes.
      if (lhs_st.getRank() < 1) return cost;
      if (matmul_op.getTransposeLhs()) {
        if (lhs_st.getRank() < 2) {
          LOG(ERROR) << "transpose_lhs=true requires lhs rank >= 2 when "
                        "dimension_numbers is absent.";
          return cost;
        }
        // If transposed, assume second to last dim is contracting.
        contracting_size_prod = lhs_st.getShape()[lhs_st.getRank() - 2];
      } else {
        // If not transposed, assume last dim is contracting.
        contracting_size_prod = lhs_st.getShape()[lhs_st.getRank() - 1];
      }
    }

    // MATMUL Computation : [M,N] [N,K] -> [M,K] :
    // Flops = M * N * K [ Multiplication ] + M * N-1 * K [ Addition ]
    // + M * K [ Accumulate Vector Addition ] = 2 * M * N * K
    cost.flops = 2 * result_st.getNumElements() * contracting_size_prod;

    const auto& data_layout = mlir::DataLayout::closest(op);
    cost.bytes_consumed = data_layout.getTypeSize(lhs_st) +
                          data_layout.getTypeSize(rhs_st) +
                          data_layout.getTypeSize(acc_st) +
                          data_layout.getTypeSize(result_st);
    return cost;
  }
};

// Estimator for multi-reduction operations.
class MultiReductionOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    const auto& data_layout = mlir::DataLayout::closest(op);
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
      if (auto shaped_type =
              mlir::dyn_cast<mlir::ShapedType>(operand.getType())) {
        if (shaped_type.hasStaticShape()) {
          cost.flops += shaped_type.getNumElements();
        }
      }
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for store operations [Write Data to Memory
class StoreOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    if (op->getNumOperands() == 0) {
      return cost;
    }
    mlir::Value input = op->getOperand(0);
    const auto& data_layout = mlir::DataLayout::closest(op);
    cost.bytes_consumed = data_layout.getTypeSize(input.getType());
    return cost;
  }
};

// Estimator for mhlo.reduce operation.
class MhloReduceOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    auto reduce_op = mlir::dyn_cast<mlir::mhlo::ReduceOp>(op);
    if (!reduce_op) {
      return cost;
    }
    const auto& data_layout = mlir::DataLayout::closest(op);
    for (mlir::Value input : reduce_op.getInputs()) {
      cost.bytes_consumed += data_layout.getTypeSize(input.getType());
      auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(input.getType());
      if (!shaped_type) {
        cost.flops += 1;  // Scalar type
      } else if (shaped_type.hasStaticShape()) {
        cost.flops += shaped_type.getNumElements();
      }
    }
    for (mlir::Value result : reduce_op.getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for transpose operations.
class TransposeOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    mlir::tpu::TransposeOp transpose_op =
        llvm::cast<mlir::tpu::TransposeOp>(op);
    if (!transpose_op) {
      return cost;
    }
    // Transpose involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    cost.bytes_consumed =
        ( data_layout.getTypeSize(transpose_op.getSourceVectorType()) +
        data_layout.getTypeSize(transpose_op.getResultVectorType()) );
    return cost;
  }
};

// Estimator for rotate operations.
class RotateOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // Rotate involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    cost.bytes_consumed =
        ( data_layout.getTypeSize(op->getOperand(0).getType()) +
        data_layout.getTypeSize(op->getResult(0).getType()) );
    return cost;
  }
};

// Estimator for concatenate operations.
class ConcatenateOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for broadcast operations.
class BroadcastOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // Broadcast involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for gather operations.
class GatherOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // Gather involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for extract operations.
class ExtractOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // Extract involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for extract strided slice operations.
class ExtractStridedSliceOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // Extract strided slice involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for PRNG random bits operations.
class PrngRandomBitsOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // PRNG random bits involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for relayout operations.
class RelayoutOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    // Relayout involves no arithmetic ops.
    cost.flops = 0;
    const auto& data_layout = mlir::DataLayout::closest(op);
    // To Estimate Memory Bandwidth, we consider the size of the inputs
    // (Reading) and outputs (Writing)
    for (mlir::Value operand : op->getOperands()) {
      cost.bytes_consumed += data_layout.getTypeSize(operand.getType());
    }
    for (mlir::Value result : op->getResults()) {
      cost.bytes_consumed += data_layout.getTypeSize(result.getType());
    }
    return cost;
  }
};

// Estimator for vector.contract operations.
class VectorContractOpEstimator : public OperationCostEstimator {
 public:
  OperationCost Estimate(mlir::Operation* op) const override {
    OperationCost cost;
    mlir::vector::ContractionOp contract_op =
        llvm::cast<mlir::vector::ContractionOp>(op);

    auto lhs_st = llvm::cast<mlir::ShapedType>(contract_op.getLhsType());
    auto rhs_st = llvm::cast<mlir::ShapedType>(contract_op.getRhsType());
    auto acc_st = llvm::cast<mlir::ShapedType>(contract_op.getAccType());
    auto result_st = llvm::cast<mlir::ShapedType>(contract_op.getResultType());

    if ( !lhs_st || !lhs_st.hasStaticShape() || !rhs_st ||
        !rhs_st.hasStaticShape() || !acc_st || !acc_st.hasStaticShape() ||
          !result_st || !result_st.hasStaticShape()) {
      return cost;
    }

    uint64_t contracting_size_prod = 1;
    auto iterator_types = contract_op.getIteratorTypes();
    auto maps = contract_op.getIndexingMapsArray();
    std::vector<int64_t> dim_sizes(iterator_types.size(), -1);

    for (unsigned i = 0; i < maps.size(); ++i) {
      mlir::AffineMap map = maps[i];
      auto input_shape = llvm::cast<mlir::ShapedType>(
          contract_op.getOperand(i).getType()).getShape();
      for (unsigned j = 0; j < map.getNumResults(); ++j) {
        if (auto dim_expr =
             llvm::dyn_cast<mlir::AffineDimExpr>(map.getResult(j))) {
          unsigned dim = dim_expr.getPosition();
          if (dim_sizes[dim] == -1) {
            dim_sizes[dim] = input_shape[j];
          } else if (dim_sizes[dim] != input_shape[j]) {
            LOG(ERROR) << "Inconsistent dim size for dim " << dim
                       << " derived from operands of vector.contract";
            return cost;  // Inconsistent dimension size
          }
        } else {
          // If map is not a simple dim expr, we can't easily get dim size.
          LOG(INFO)
              << "Cannot estimate cost for vector.contract with complex maps";
          return cost;
        }
      }
    }

    for (unsigned i = 0; i < iterator_types.size(); ++i) {
      auto cast_val = llvm::cast<mlir::vector::IteratorTypeAttr>(
          iterator_types[i]).getValue();
      if (cast_val == mlir::vector::IteratorType::reduction) {
        if (dim_sizes[i] == -1) {
          LOG(ERROR) << "Could not determine size of reduction dimension " << i
                     << " for vector.contract";
          return cost;
        }
        contracting_size_prod *= dim_sizes[i];
      }
    }

    cost.flops = 2 * result_st.getNumElements() * contracting_size_prod;

    const auto& data_layout = mlir::DataLayout::closest(op);
    cost.bytes_consumed = data_layout.getTypeSize(lhs_st) +
                          data_layout.getTypeSize(rhs_st) +
                          data_layout.getTypeSize(acc_st) +
                          data_layout.getTypeSize(result_st);
    return cost;
  }
};

// Hasher for mlir::TypeID to be used with absl::flat_hash_map.
struct MlirTypeIDHasher {
  size_t operator()(mlir::TypeID id) const { return mlir::hash_value(id); }
};

class CostModel {
 public:
  static const CostModel& GetInstance() {
    static absl::NoDestructor<CostModel> instance;
    return *instance;
  }

  OperationCost GetOperationCost(mlir::Operation* op) const {
    auto it = estimators_.find(op->getName().getTypeID());
    if (it != estimators_.end()) {
      return it->second->Estimate(op);
    }
    return {};
  }

 private:
  friend class absl::NoDestructor<CostModel>;
  CostModel() {
    RegisterEstimator<ElementWiseOpEstimator>(
        {mlir::TypeID::get<mlir::arith::CmpIOp>(),
         mlir::TypeID::get<mlir::arith::ExtUIOp>(),
         mlir::TypeID::get<mlir::arith::MulIOp>(),
         mlir::TypeID::get<mlir::arith::IndexCastOp>(),
         mlir::TypeID::get<mlir::arith::MaximumFOp>(),
         mlir::TypeID::get<mlir::arith::SubFOp>(),
         mlir::TypeID::get<mlir::math::ExpOp>(),
         mlir::TypeID::get<mlir::arith::AddFOp>(),
         mlir::TypeID::get<mlir::arith::DivFOp>(),
         mlir::TypeID::get<mlir::arith::CmpFOp>(),
         mlir::TypeID::get<mlir::arith::SelectOp>(),
         mlir::TypeID::get<mlir::arith::MulFOp>(),
         mlir::TypeID::get<mlir::arith::TruncFOp>(),
         mlir::TypeID::get<mlir::arith::ExtSIOp>(),
         mlir::TypeID::get<mlir::arith::TruncIOp>(),
         mlir::TypeID::get<mlir::tpu::IotaOp>()});

    RegisterEstimator<MemoryOnlyOpEstimator>(
        {mlir::TypeID::get<mlir::arith::ConstantOp>(),
         mlir::TypeID::get<mlir::vector::LoadOp>(),
         mlir::TypeID::get<mlir::tpu::LoadOp>(),
         mlir::TypeID::get<mlir::memref::LoadOp>(),
         mlir::TypeID::get<mlir::tpu::StridedLoadOp>(),
         mlir::TypeID::get<mlir::tpu::BitcastOp>()});

    RegisterEstimator<MatmulOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::MatmulOp>()});

    RegisterEstimator<MultiReductionOpEstimator>(
        {mlir::TypeID::get<mlir::vector::ReductionOp>(),
         mlir::TypeID::get<mlir::vector::MultiDimReductionOp>(),
         mlir::TypeID::get<mlir::tpu::ReduceIndexOp>()});

    RegisterEstimator<MhloReduceOpEstimator>(
        {mlir::TypeID::get<mlir::mhlo::ReduceOp>()});

    RegisterEstimator<StoreOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::StoreOp>(),
         mlir::TypeID::get<mlir::vector::StoreOp>(),
         mlir::TypeID::get<mlir::tpu::StridedStoreOp>()});

    RegisterEstimator<TransposeOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::TransposeOp>()});

    RegisterEstimator<RotateOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::RotateOp>(),
         mlir::TypeID::get<mlir::tpu::DynamicRotateOp>()});

    RegisterEstimator<ConcatenateOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::ConcatenateOp>()});

    RegisterEstimator<BroadcastOpEstimator>(
        {mlir::TypeID::get<mlir::vector::BroadcastOp>()});

    RegisterEstimator<GatherOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::GatherOp>(),
         mlir::TypeID::get<mlir::tpu::DynamicGatherOp>()});

    RegisterEstimator<ExtractOpEstimator>(
        {mlir::TypeID::get<mlir::vector::ExtractOp>()});

    RegisterEstimator<ExtractStridedSliceOpEstimator>(
        {mlir::TypeID::get<mlir::vector::ExtractStridedSliceOp>()});

    RegisterEstimator<PrngRandomBitsOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::PRNGRandomBitsOp>()});

    RegisterEstimator<RelayoutOpEstimator>(
        {mlir::TypeID::get<mlir::tpu::RelayoutOp>()});

    RegisterEstimator<VectorContractOpEstimator>(
        {mlir::TypeID::get<mlir::vector::ContractionOp>()});
  }

  template <typename T>
  void RegisterEstimator(std::initializer_list<mlir::TypeID> op_type_ids) {
    auto estimator = std::make_unique<T>();
    for (const auto& op_type_id : op_type_ids) {
      estimators_[op_type_id] = estimator.get();
    }
    owned_estimators_.push_back(std::move(estimator));
  }

  absl::flat_hash_map<mlir::TypeID, const OperationCostEstimator*,
                      MlirTypeIDHasher>
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
