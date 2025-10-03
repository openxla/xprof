#include "xprof/utils/custom_call_utils.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"
#include "xprof/utils/backend_configs.pb.h"

namespace xprof {

absl::StatusOr<std::string> GetCustomCallText(
    const xla::HloInstruction& hlo_instruction) {
  if (!hlo_instruction.has_backend_config()) {
    return absl::NotFoundError("Backend config not found");
  }
  tsl::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  BackendConfig config;
  TF_RETURN_IF_ERROR(tsl::protobuf::util::JsonStringToMessage(
      hlo_instruction.raw_backend_config_string(), &config, options));
  if (!config.has_custom_call_config()) {
    return absl::NotFoundError("Custom call config not found");
  }
  CustomCallConfig custom_call_config = config.custom_call_config();
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(true);
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_op,
      xla::ParseMlirModuleString(
          static_cast<std::string>(custom_call_config.body()), context));
  return xla::llvm_ir::DumpToString(*mlir_op);
}

}  // namespace xprof
