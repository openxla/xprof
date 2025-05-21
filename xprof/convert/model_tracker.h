#ifndef THIRD_PARTY_XPROF_CONVERT_MODEL_TRACKER_H_
#define THIRD_PARTY_XPROF_CONVERT_MODEL_TRACKER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xprof/utils/hlo_module_map.h"

namespace tensorflow::profiler {

class ModelTracker {
 public:
  virtual ~ModelTracker() = default;

  enum Framework {
    kUnknown,
    kTensorFlow,   // has TensorFlow ops
    kTensorFlow1,  // has "gradients"
    kTensorFlow2,  // has "gradient_tape"
    kJax,
  };

  // If <return_on_training> is set to true, this function will return
  // immediately when it finds the first HLO instruction related to training.
  void ProcessHloProto(const xla::HloProto& hlo_proto,
                       bool return_on_training = false,
                       bool return_on_barnacore = false);

  template <class T>
  void ProcessHloModule(const HloModuleInterface<T>& hlo_module,
                        bool return_on_training = false,
                        bool return_on_barnacore = false) {
    DCHECK(!(return_on_training && return_on_barnacore));
    for (const auto* instr : hlo_module.Instructions()) {
      ProcessInstructionMetadata(*instr);
      if (return_on_training && IsTraining()) return;
      if (return_on_barnacore && UsesBarnaCore()) return;
    }
  }

  void ProcessOp(absl::string_view op_name, absl::string_view op_type);
  virtual void ProcessXlaOpCategory(absl::string_view op_category);

  bool IsTraining() const;

  Framework GetFramework() const;

  std::string GetTaskType() const;

  std::string GetArchitectureType() const;

  bool HasBertTfOp() const { return has_bert_tf_op_; }
  bool HasLambTfOp() const { return has_lamb_tf_op_; }

  bool UsesBarnaCore() const { return has_barna_core_op_; }
  bool HasSendRecvOps() const { return has_send_recv_op_; }

  bool HasGradients() const;
  bool HasAllReduce() const { return has_all_reduce_op_; }

 protected:
  void ProcessInstructionMetadata(const HloInstructionInterface& instr);
  void ProcessOpName(absl::string_view op_name);
  virtual void ProcessTfOp(absl::string_view op_name,
                           absl::string_view op_type);
  virtual void ProcessJaxOp(absl::string_view op_name,
                            absl::string_view op_type = "");

  // TensorFlow
  bool has_tf_op_ = false;
  bool has_tf_gradients_op_ = false;
  bool has_tf_gradient_tape_op_ = false;

  bool has_rnn_cell_ = false;
  bool has_lstm_cell_ = false;
  bool has_gru_cell_ = false;
  bool has_conv_op_ = false;
  bool has_matmul_op_ = false;

  bool has_bert_tf_op_ = false;
  bool has_lamb_tf_op_ = false;

  // JAX
  bool has_jax_op_ = false;
  bool has_jax_transpose_namescope_ = false;

  // XLA
  bool has_all_reduce_op_ = false;
  bool has_barna_core_op_ = false;
  bool has_send_recv_op_ = false;
};

}  // namespace tensorflow::profiler

#endif  // THIRD_PARTY_XPROF_CONVERT_MODEL_TRACKER_H_
