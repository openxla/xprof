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

#ifndef THIRD_PARTY_XPROF_CONVERT_FILE_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_FILE_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/cloud/storage/client.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "tsl/platform/protobuf.h"

namespace xprof {

// Reads a binary proto from a GCS path (gs://bucket/object) using the
// google-cloud-cpp storage API.
absl::Status ReadBinaryProto(const std::string& fname,
                             tsl::protobuf::MessageLite* proto);

// Writes a binary proto to a GCS path (gs://bucket/object) using the
// google-cloud-cpp storage API.
// Falls back to tsl::WriteBinaryProto for non-GCS paths.
absl::Status WriteBinaryProto(const std::string& fname,
                              const tsl::protobuf::MessageLite& proto);

namespace internal {

// Parses a GCS path. Supports gs:// and /bigstore/ prefixes.
absl::Status ParseGcsPath(absl::string_view fname, std::string* bucket,
                          std::string* object);

// Internal implementation that takes a GCS client, used for testing.
absl::Status ReadBinaryProtoWithClient(google::cloud::storage::Client& client,
                                       const std::string& fname,
                                       tsl::protobuf::MessageLite* proto);

// Internal implementation that takes a GCS client, used for testing.
absl::Status WriteBinaryProtoWithClient(
    google::cloud::storage::Client& client, const std::string& fname,
    const tsl::protobuf::MessageLite& proto);

}  // namespace internal

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_FILE_UTILS_H_
