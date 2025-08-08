/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_HOST_OP_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_HOST_OP_UTILS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
namespace tensorflow {
namespace profiler {

// Classification of input processing on the host.
enum class InputOpCategory {
  kEnqueue,        // enqueue data to be transferred to device.
  kRead,           // read from file
  kPreprocessing,  // data preprocessing.
  kUnknown,        // unknown category.
};

inline std::string InputOpCategoryString(InputOpCategory category) {
  switch (category) {
    case InputOpCategory::kEnqueue:
      return "Enqueue";
    case InputOpCategory::kRead:
      return "Read";
    case InputOpCategory::kPreprocessing:
      return "Preprocessing";
    case InputOpCategory::kUnknown:
      return "Unknown";
  }
}

// Type of the host op (where the traceme is collected)
enum HostOpType {
  kTfOp,
  kJaxOp,
  kPyGrainOp,
  kUnknown,
};

inline std::string HostOpTypeToString(HostOpType type) {
  switch (type) {
    case HostOpType::kPyGrainOp:
      return "PyGrainOp";
    case HostOpType::kTfOp:
      return "TfOp";
    case HostOpType::kJaxOp:
      return "JaxOp";
    default:
      return "Unknown";
  }
}

// Struct for an input pipeline op.
struct HostOp {
  HostOpType type = HostOpType::kUnknown;
  std::string display_name;  // name shows on trace viewer UI.
  // name should be unique and representative of underlying functions.
  std::string name;
  // for input pipeline ops, it is the category of the stage.
  // use string instead of enum to be backward compatible.
  std::string category;
  // key value pairs for host ops.
  absl::flat_hash_map<std::string, std::string> args;

  void set_name(absl::string_view name) { this->name = name; }
  void set_category(absl::string_view category) { this->category = category; }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_HOST_OP_UTILS_H_
