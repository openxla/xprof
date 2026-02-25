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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/cloud/options.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/client.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/download_options.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/options.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/protobuf.h"

namespace xprof {

namespace {

namespace gcs = ::google::cloud::storage;

// Number of concurrent threads for downloading.
constexpr int kNumThreads = 32;
// Minimum chunk size to justify parallelization (8 MB).
constexpr int64_t kMinChunkSize = 8 * 1024 * 1024;
// Maximum size for a proto, roughly 2GB.
constexpr int64_t kMaxProtoSize = 2LL * 1024 * 1024 * 1024;

tsl::thread::ThreadPool* GetGcsThreadPool() {
  static tsl::thread::ThreadPool* pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), "gcs_read", kNumThreads);
  return pool;
}

gcs::Client& GetGcsClient() {
  static auto* client = []() {
    auto options = google::cloud::Options{}
                       .set<gcs::ConnectionPoolSizeOption>(kNumThreads)
                       .set<gcs::DownloadBufferSizeOption>(1024 * 1024);
    return new gcs::Client(std::move(options));
  }();
  return *client;
}

}  // namespace
namespace internal {

absl::Status ParseGcsPath(absl::string_view fname, std::string* bucket,
                          std::string* object) {
  absl::string_view path = fname;
  const std::string gcs_prefix = "gs://";
  const std::string bigstore_prefix = "/bigstore/";

  if (absl::StartsWith(path, gcs_prefix)) {
    path = absl::StripPrefix(path, gcs_prefix);
  } else if (absl::StartsWith(path, bigstore_prefix)) {
    path = absl::StripPrefix(path, bigstore_prefix);
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "GCS path must start with 'gs://' or '/bigstore/': ", fname));
  }

  size_t slash_pos = path.find('/');
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

}  // namespace internal
namespace {

absl::Status DownloadConcurrently(gcs::Client& client,
                                  const std::string& bucket,
                                  const std::string& object, int64_t total_size,
                                  std::string& contents) {
  contents.resize(total_size);

  int64_t chunk_size =
      std::max(kMinChunkSize, (total_size + kNumThreads - 1) / kNumThreads);
  int num_chunks = (total_size + chunk_size - 1) / chunk_size;

  absl::Mutex mu;
  absl::Status status = absl::OkStatus();

  GetGcsThreadPool()->ParallelFor(
      num_chunks, tsl::thread::ThreadPool::SchedulingParams::Fixed(1),
      [&client, &bucket, &object, chunk_size, total_size, &contents, &mu,
       &status](int64_t i, int64_t end_chunk) {
        for (int64_t chunk_idx = i; chunk_idx < end_chunk; ++chunk_idx) {
          {
            absl::MutexLock lock(mu);
            if (!status.ok()) return;
          }
          int64_t start = chunk_idx * chunk_size;
          int64_t end = std::min(start + chunk_size, total_size);

          auto reader =
              client.ReadObject(bucket, object, gcs::ReadRange(start, end));
          if (!reader) {
            absl::MutexLock lock(mu);
            status.Update(absl::InternalError(absl::StrCat(
                "Failed to read range: ", reader.status().message())));
            return;
          }
          reader.read(&contents[start], end - start);
          if (!reader.status().ok()) {
            absl::MutexLock lock(mu);
            status.Update(absl::DataLossError(absl::StrCat(
                "Failed to read GCS range data: ", reader.status().message())));
            return;
          }
        }
      });
  return status;
}

}  // namespace
namespace internal {

absl::Status ReadBinaryProtoWithClient(gcs::Client& client,
                                       const std::string& fname,
                                       tsl::protobuf::MessageLite* proto) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));

  // Get object metadata to find the size.
  auto metadata = client.GetObjectMetadata(bucket, object);
  if (!metadata) {
    return absl::NotFoundError(absl::StrCat("Failed to get GCS metadata: ",
                                            metadata.status().message()));
  }

  int64_t total_size = static_cast<int64_t>(metadata->size());
  if (total_size == 0) {
    proto->Clear();
    return absl::OkStatus();
  }

  if (total_size > kMaxProtoSize) {
    return absl::FailedPreconditionError(
        absl::StrCat("File too large for a proto: ", total_size));
  }

  std::string contents;
  absl::Time start_download = absl::Now();
  TF_RETURN_IF_ERROR(
      DownloadConcurrently(client, bucket, object, total_size, contents));
  absl::Time end_download = absl::Now();
  VLOG(1) << "Download from GCS took: " << end_download - start_download;

  absl::Time start_parse = absl::Now();
  if (!proto->ParseFromString(contents)) {
    return absl::DataLossError(
        absl::StrCat("Can't parse ", fname, " as binary proto"));
  }
  absl::Time end_parse = absl::Now();
  VLOG(1) << "Protobuf parsing took: " << end_parse - start_parse;

  return absl::OkStatus();
}

absl::Status WriteBinaryProtoWithClient(
    gcs::Client& client, const std::string& fname,
    const tsl::protobuf::MessageLite& proto) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseGcsPath(fname, &bucket, &object));

  std::string contents;
  absl::Time start_serialize = absl::Now();
  if (!proto.SerializeToString(&contents)) {
    return absl::InternalError(
        absl::StrCat("Failed to serialize proto to string for ", fname));
  }
  absl::Time end_serialize = absl::Now();
  LOG(INFO) << "Proto serialization took: " << end_serialize - start_serialize;

  absl::Time start_upload = absl::Now();
  auto stream = client.WriteObject(bucket, object);
  stream << contents;
  stream.Close();
  absl::Time end_upload = absl::Now();
  if (!stream) {
    return absl::InternalError(absl::StrCat(
        "Failed to write to GCS: ", stream.metadata().status().message()));
  }
  LOG(INFO) << "Upload to GCS took: " << end_upload - start_upload;
  return absl::OkStatus();
}

}  // namespace internal

absl::Status ReadBinaryProto(const std::string& fname,
                             tsl::protobuf::MessageLite* proto) {
  if (absl::StartsWith(fname, "gs://") ||
      absl::StartsWith(fname, "/bigstore/")) {
    return internal::ReadBinaryProtoWithClient(GetGcsClient(), fname, proto);
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
    return internal::WriteBinaryProtoWithClient(GetGcsClient(), gcs_path,
                                                proto);
  }

  return tsl::WriteBinaryProto(tsl::Env::Default(), fname, proto);
}

}  // namespace xprof
