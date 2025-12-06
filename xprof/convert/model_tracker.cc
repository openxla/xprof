#include "xprof/convert/model_tracker.h"

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xprof/utils/hlo_module_map.h"

namespace tensorflow::profiler {
namespace {

using ::tsl::profiler::IsJaxOpNameAndType;
using ::tsl::profiler::IsTfOpName;
using ::tsl::profiler::IsTfOpType;
using ::tsl::profiler::ParseTfNameScopes;

bool IsBertTfOp(absl::string_view op_name) {
  return absl::StrContains(op_name, "bert") ||
         absl::StrContains(op_name, "BERT");
}

bool IsLambTfOp(absl::string_view op_name) {
  return absl::StrContains(op_name, "LAMB");
}

}  // namespace

void ModelTracker::ProcessTfOp(absl::string_view op_name,
                               absl::string_view op_type) {
  has_tf_op_ = true;

  if (!has_tf_gradient_tape_op_ || !has_tf_gradients_op_) {
    for (absl::string_view scope : ParseTfNameScopes(op_name)) {
      if (absl::ConsumePrefix(&scope, "gradient")) {
        if (scope == "_tape") {
          // matched "gradient_tape"
          has_tf_gradient_tape_op_ = true;
        } else if (absl::StartsWith(scope, "s")) {
          // matched "gradients", "gradients_2", "gradients_3", etc.
          has_tf_gradients_op_ = true;
        }
      }
    }
  }
}

void ModelTracker::ProcessJaxOp(absl::string_view op_name,
                                absl::string_view op_type) {
  has_jax_op_ = true;

  // b/224774734#comment3: There aren't enough RNN/LSTM models in JAX for it
  // to be worth trying to identify or pattern match them.

  if (!has_jax_transpose_namescope_) {
    for (absl::string_view name_scope : absl::StrSplit(op_name, '/')) {
      // There are two kinds of transpose in JAX namescope.
      // 1. transpose(jvp(...)) which is part of the gradient computation.
      // 2. transpose[permutation=(...)], which is just plain tensor
      //    transposition.
      // We want to limit our gradient checks to only the first case.
      if (absl::StartsWith(name_scope, "transpose(")) {
        has_jax_transpose_namescope_ = true;
        break;
      }
    }
  }
}

void ModelTracker::ProcessOpName(absl::string_view op_name) {
  if (!has_bert_tf_op_ && IsBertTfOp(op_name)) {
    has_bert_tf_op_ = true;
  }
  if (!has_lamb_tf_op_ && IsLambTfOp(op_name)) {
    has_lamb_tf_op_ = true;
  }
}

void ModelTracker::ProcessXlaOpCategory(absl::string_view category) {
  if (!has_all_reduce_op_ && tsl::profiler::IsAllReduceOrAllToAll(category)) {
    has_all_reduce_op_ = true;
  }
}

bool ModelTracker::IsTraining() const { return HasGradients(); }

bool ModelTracker::HasGradients() const {
  return has_tf_gradients_op_ || has_tf_gradient_tape_op_ ||
         has_jax_transpose_namescope_;
}

ModelTracker::Framework ModelTracker::GetFramework() const {
  if (has_tf_gradient_tape_op_) return Framework::kTensorFlow2;
  if (has_tf_gradients_op_) return Framework::kTensorFlow1;
  if (has_tf_op_) return Framework::kTensorFlow;
  if (has_jax_op_) return Framework::kJax;
  return Framework::kUnknown;
}

std::string ModelTracker::GetTaskType() const {
  return IsTraining() ? "Training" : "Inference";
}

std::string ModelTracker::GetArchitectureType() const {
  if (has_lstm_cell_) return "Long Short Term Memory (LSTM)";
  if (has_gru_cell_) return "Gated Recurrent Unit (GRU)";
  if (has_rnn_cell_) return "Recurrent Neural Network (RNN)";
  if (has_conv_op_) return "Convolutional Neural Network (CNN)";
  if (has_matmul_op_) return "Multi-Layer Perceptron (MLP)";
  return "Unknown";
}

void ModelTracker::ProcessInstructionMetadata(
    const HloInstructionInterface& instr) {
  absl::string_view op_name = instr.Metadata().op_name();
  absl::string_view op_type = instr.Metadata().op_type();
  ProcessOp(op_name, op_type);
  ProcessOpName(op_name);
  ProcessXlaOpCategory(instr.Category());
}

void ModelTracker::ProcessOp(absl::string_view op_name,
                             absl::string_view op_type) {
  if (IsTfOpType(op_type) && IsTfOpName(op_name)) {
    ProcessTfOp(op_name, op_type);
  } else if (IsJaxOpNameAndType(op_name, op_type)) {
    ProcessJaxOp(op_name, op_type);
  } else if (op_type.empty()) {
    // (b/230753633) op_type are missing for a portion of JAX HLOs.
    // Process op_name only.
    ProcessJaxOp(op_name);
  }
}

void ModelTracker::ProcessHloProto(const xla::HloProto& hlo_proto,
                                   bool return_on_training,
                                   bool return_on_scv0) {
  HloModuleWrapper hlo_module(hlo_proto);
  ProcessHloModule(hlo_module, return_on_training, return_on_scv0);
}

}  // namespace tensorflow::profiler
