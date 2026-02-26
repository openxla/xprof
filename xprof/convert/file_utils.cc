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

#include "xprof/convert/file_utils.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/file_utils_internal.h"
#include "xprof/convert/storage_client_interface.h"

namespace xprof {

namespace {

// Maximum size for a proto, roughly 2GB.
constexpr int64_t kMaxProtoSize = 2LL * 1024 * 1024 * 1024;

}  // namespace

namespace internal {

absl::Status ParseGcsPath(absl::string_view fname, std::string* bucket,
                          std::string* object) {
  absl::string_view path = fname;
  constexpr absl::string_view kGcsPrefix = "gs://";
  constexpr absl::string_view kBigstorePrefix = "/bigstore/";

  if (absl::StartsWith(path, kGcsPrefix)) {
    path = absl::StripPrefix(path, kGcsPrefix);
  } else if (absl::StartsWith(path, kBigstorePrefix)) {
    path = absl::StripPrefix(path, kBigstorePrefix);
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "GCS path must start with 'gs://' or '/bigstore/': ", fname));
  }

  const size_t slash_pos = path.find('/');
  if (slash_pos == absl::string_view::npos || slash_pos == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("GCS path doesn't contain a bucket name: ", fname));
  }
  *bucket = std::string(path.substr(0, slash_pos));
  *object = std::string(path.substr(slash_pos + 1));
  if (object->empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("GCS path doesn't contain an object name: ", fname));
  }
  return absl::OkStatus();
}

absl::Status ReadBinaryProtoWithClient(StorageClientInterface& client,
                                       const std::string& fname,
                                       tsl::protobuf::MessageLite* proto) {
  std::string bucket;
  std::string object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));

  // Get object size.
  const absl::StatusOr<std::uint64_t> size_or =
      client.GetObjectSize(bucket, object);
  if (!size_or.ok()) {
    return absl::NotFoundError(absl::StrCat("Failed to get GCS metadata: ",
                                            size_or.status().message()));
  }

  const std::uint64_t total_size = *size_or;
  if (total_size == 0) {
    proto->Clear();
    return absl::OkStatus();
  }

  if (total_size > static_cast<std::uint64_t>(kMaxProtoSize)) {
    return absl::FailedPreconditionError(
        absl::StrCat("File too large for a proto: ", total_size));
  }

  std::string contents;
  contents.resize(total_size);
  const absl::Time start_download = absl::Now();
  TF_RETURN_IF_ERROR(
      client.ReadObject(bucket, object, 0, total_size, &contents[0]));
  const absl::Time end_download = absl::Now();
  VLOG(1) << "Download from GCS took: " << end_download - start_download;

  const absl::Time start_parse = absl::Now();
  if (!proto->ParseFromString(contents)) {
    return absl::DataLossError(
        absl::StrCat("Can't parse ", fname, " as binary proto"));
  }
  const absl::Time end_parse = absl::Now();
  VLOG(1) << "Protobuf parsing took: " << end_parse - start_parse;

  return absl::OkStatus();
}

absl::Status WriteBinaryProtoWithClient(
    StorageClientInterface& client, const std::string& fname,
    const tsl::protobuf::MessageLite& proto) {
  std::string bucket;
  std::string object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));

  std::string contents;
  const absl::Time start_serialize = absl::Now();
  if (!proto.SerializeToString(&contents)) {
    return absl::InternalError(
        absl::StrCat("Failed to serialize proto to string for ", fname));
  }
  const absl::Time end_serialize = absl::Now();
  LOG(INFO) << "Proto serialization took: " << end_serialize - start_serialize;

  const absl::Time start_upload = absl::Now();
  TF_RETURN_IF_ERROR(client.WriteObject(bucket, object, contents));
  const absl::Time end_upload = absl::Now();
  LOG(INFO) << "Upload to GCS took: " << end_upload - start_upload;
  return absl::OkStatus();
}

}  // namespace internal

absl::Status ReadBinaryProto(const std::string& fname,
                             tsl::protobuf::MessageLite* proto) {
  if (absl::StartsWith(fname, "gs://") ||
      absl::StartsWith(fname, "/bigstore/")) {
    return internal::ReadBinaryProtoWithClient(internal::GetDefaultGcsClient(),
                                               fname, proto);
  }

  return tsl::ReadBinaryProto(tsl::Env::Default(), fname, proto);
}

absl::Status WriteBinaryProto(const std::string& fname,
                              const tsl::protobuf::MessageLite& proto) {
  if (absl::StartsWith(fname, "gs://") ||
      absl::StartsWith(fname, "/bigstore/")) {
    std::string gcs_path = fname;
    if (absl::StartsWith(fname, "/bigstore/")) {
      gcs_path = absl::StrCat("gs://", absl::StripPrefix(fname, "/bigstore/"));
    }
    return internal::WriteBinaryProtoWithClient(internal::GetDefaultGcsClient(),
                                                gcs_path, proto);
  }

  return tsl::WriteBinaryProto(tsl::Env::Default(), fname, proto);
}

}  // namespace xprof
