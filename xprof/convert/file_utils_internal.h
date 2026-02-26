/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_FILE_UTILS_INTERNAL_H_
#define THIRD_PARTY_XPROF_CONVERT_FILE_UTILS_INTERNAL_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/storage_client_interface.h"

namespace xprof {
namespace internal {

// Parses a GCS path. Supports gs:// and /bigstore/ prefixes.
absl::Status ParseGcsPath(absl::string_view fname, std::string* bucket,
                          std::string* object);

// Internal implementation that takes a storage client interface.
absl::Status ReadBinaryProtoWithClient(StorageClientInterface& client,
                                       const std::string& fname,
                                       tsl::protobuf::MessageLite* proto);

// Internal implementation that takes a storage client interface.
absl::Status WriteBinaryProtoWithClient(
    StorageClientInterface& client, const std::string& fname,
    const tsl::protobuf::MessageLite& proto);

// Returns the default GCS client implementation. Defined in file_utils_gcs.cc.
StorageClientInterface& GetDefaultGcsClient();

}  // namespace internal
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_FILE_UTILS_INTERNAL_H_
