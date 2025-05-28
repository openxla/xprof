/* Copyright 2025 The XProf Authors. All Rights Reserved.

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

#include "xprof/utils/tensorflow_utils.h"
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/strcat.h"
#include "plugin/xprof/protobuf/tensorflow_datatypes.pb.h"

namespace tensorflow {
namespace profiler {

absl::Status ParseTextFormatFromString(std::string input,
                                              google::protobuf::Message* output) {
  if (output == nullptr) {
    return absl::InvalidArgumentError("Output proto message must be non-NULL.");
  }
  google::protobuf::TextFormat::Parser parser;
  std::string error_msg;
  if (!parser.ParseFromString(input, output)) {
    return absl::InvalidArgumentError(error_msg);
  }
  return absl::OkStatus();
}

tsl::string DataTypeString(TensorflowDataType dtype) {
  if (IsRefType(dtype)) {
    TensorflowDataType non_ref = static_cast<TensorflowDataType>(
        static_cast<int>(dtype) - static_cast<int>(kDataTypeRefOffset));
    return tsl::strings::StrCat(DataTypeStringInternal(non_ref), "_ref");
  }
  return DataTypeStringInternal(dtype);
}

tsl::string DataTypeStringInternal(TensorflowDataType dtype) {
  switch (dtype) {
    case DT_INVALID:
      return "INVALID";
    case DT_FLOAT:
      return "float";
    case DT_DOUBLE:
      return "double";
    case DT_INT32:
      return "int32";
    case DT_UINT32:
      return "uint32";
    case DT_UINT8:
      return "uint8";
    case DT_UINT16:
      return "uint16";
    case DT_INT16:
      return "int16";
    case DT_INT8:
      return "int8";
    case DT_STRING:
      return "string";
    case DT_COMPLEX64:
      return "complex64";
    case DT_COMPLEX128:
      return "complex128";
    case DT_INT64:
      return "int64";
    case DT_UINT64:
      return "uint64";
    case DT_BOOL:
      return "bool";
    case DT_QINT8:
      return "qint8";
    case DT_QUINT8:
      return "quint8";
    case DT_QUINT16:
      return "quint16";
    case DT_QINT16:
      return "qint16";
    case DT_QINT32:
      return "qint32";
    case DT_BFLOAT16:
      return "bfloat16";
    case DT_HALF:
      return "half";
    case DT_FLOAT8_E5M2:
      return "float8_e5m2";
    case DT_FLOAT8_E4M3FN:
      return "float8_e4m3fn";
    case DT_FLOAT8_E4M3FNUZ:
      return "float8_e4m3fnuz";
    case DT_FLOAT8_E4M3B11FNUZ:
      return "float8_e4m3b11fnuz";
    case DT_FLOAT8_E5M2FNUZ:
      return "float8_e5m2fnuz";
    case DT_INT4:
      return "int4";
    case DT_UINT4:
      return "uint4";
    case DT_RESOURCE:
      return "resource";
    case DT_VARIANT:
      return "variant";
    default:
      LOG(ERROR) << "Unrecognized DataType enum value " << dtype;
      return tsl::strings::StrCat("unknown dtype enum (", dtype, ")");
  }
}
}  // namespace profiler
}  // namespace tensorflow
