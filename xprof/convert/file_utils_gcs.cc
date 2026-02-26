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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "google/cloud/options.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/status.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/status_or.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/client.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/download_options.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/object_metadata.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/object_read_stream.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/object_write_stream.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/options.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xprof/convert/storage_client_interface.h"

namespace xprof {
namespace {

namespace gcs = ::google::cloud::storage;

// Number of concurrent threads for downloading.
constexpr int kNumThreads = 32;
// Minimum chunk size to justify parallelization (8 MB).
constexpr int64_t kMinChunkSize = 8 * 1024 * 1024;
// Download buffer size for GCS client (1 MB).
constexpr int kDownloadBufferSize = 1024 * 1024;

tsl::thread::ThreadPool& GetGcsThreadPool() {
  static absl::NoDestructor<tsl::thread::ThreadPool> pool(
      tsl::Env::Default(), "gcs_read", kNumThreads);
  return *pool;
}

// Implementation of StorageClientInterface for GCS.
class GcsStorageClient : public internal::StorageClientInterface {
 public:
  explicit GcsStorageClient(gcs::Client client) : client_(std::move(client)) {}

  // Returns the size of the object in GCS.
  absl::StatusOr<std::uint64_t> GetObjectSize(
      const std::string& bucket, const std::string& object) override {
    const google::cloud::StatusOr<gcs::ObjectMetadata> metadata =
        client_.GetObjectMetadata(bucket, object);
    if (!metadata) {
      if (metadata.status().code() == google::cloud::StatusCode::kNotFound) {
        return absl::NotFoundError(metadata.status().message());
      }
      return absl::InternalError(metadata.status().message());
    }
    return metadata->size();
  }

  // Reads the object from GCS in parallel chunks.
  absl::Status ReadObject(const std::string& bucket, const std::string& object,
                          std::uint64_t start, std::uint64_t end,
                          char* buffer) override {
    const std::uint64_t total_size = end - start;
    const std::uint64_t chunk_size = std::max<std::uint64_t>(
        kMinChunkSize, (total_size + kNumThreads - 1) / kNumThreads);
    const int num_chunks = static_cast<int>(
        (total_size + chunk_size - 1) / chunk_size);

    absl::Mutex mu;
    absl::Status status = absl::OkStatus();

    GetGcsThreadPool().ParallelFor(
        num_chunks, tsl::thread::ThreadPool::SchedulingParams::Fixed(1),
        [this, &bucket, &object, start, chunk_size, total_size, buffer, &mu,
         &status](int64_t i, int64_t end_chunk) {
          for (int64_t chunk_idx = i; chunk_idx < end_chunk; ++chunk_idx) {
            {
              absl::MutexLock lock(&mu);
              if (!status.ok()) return;
            }
            const std::uint64_t chunk_start = start + chunk_idx * chunk_size;
            const std::uint64_t chunk_end =
                std::min(chunk_start + chunk_size, start + total_size);

            gcs::ObjectReadStream reader = client_.ReadObject(
                bucket, object, gcs::ReadRange(chunk_start, chunk_end));
            if (!reader) {
              absl::MutexLock lock(&mu);
              status.Update(absl::InternalError(absl::StrCat(
                  "Failed to read range: ", reader.status().message())));
              return;
            }
            reader.read(buffer + (chunk_start - start),
                        chunk_end - chunk_start);
            if (!reader.status().ok()) {
              absl::MutexLock lock(&mu);
              status.Update(absl::DataLossError(absl::StrCat(
                  "Failed to read GCS range data: ",
                  reader.status().message())));
              return;
            }
          }
        });
    return status;
  }

  // Writes the contents to the object in GCS.
  absl::Status WriteObject(const std::string& bucket, const std::string& object,
                           const std::string& contents) override {
    gcs::ObjectWriteStream stream = client_.WriteObject(bucket, object);
    stream << contents;
    stream.Close();
    if (!stream) {
      return absl::InternalError(absl::StrCat(
          "Failed to write to GCS: ", stream.metadata().status().message()));
    }
    return absl::OkStatus();
  }

 private:
  gcs::Client client_;
};

}  // namespace

namespace internal {

StorageClientInterface& GetDefaultGcsClient() {
  static absl::NoDestructor<GcsStorageClient> client([] {
    google::cloud::Options options =
        google::cloud::Options{}
            .set<gcs::ConnectionPoolSizeOption>(kNumThreads)
            .set<gcs::DownloadBufferSizeOption>(kDownloadBufferSize);
    return GcsStorageClient(gcs::Client(std::move(options)));
  }());
  return *client;
}

}  // namespace internal
}  // namespace xprof
