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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_TSL_ARROW_OUTPUT_STREAM_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_TSL_ARROW_OUTPUT_STREAM_H_

#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "arrow/io/interfaces.h"  // from @arrow
#include "arrow/result.h"  // from @arrow
#include "arrow/status.h"  // from @arrow
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace xprof::events_db::internal {

// An `arrow::io::OutputStream` implementation backed by a `tsl::WritableFile`.
// This enables writing Arrow and Parquet data to any filesystem supported by
// `tsl::Env` (e.g. local POSIX filesystems, Google CNS, CFS, and Bigstore/GCS).
class TslArrowOutputStream final : public arrow::io::OutputStream {
 public:
  // Opens a writable file at `file_path` using `env` and returns an output
  // stream.
  static absl::StatusOr<std::shared_ptr<TslArrowOutputStream>> Open(
      absl::string_view file_path, tsl::Env* env = nullptr);

  explicit TslArrowOutputStream(
      absl_nonnull std::unique_ptr<tsl::WritableFile> file);

  ~TslArrowOutputStream() override;

  arrow::Status Write(const void* data, int64_t nbytes) override;
  arrow::Status Flush() override;
  arrow::Status Close() override;
  arrow::Result<int64_t> Tell() const override;
  bool closed() const override { return is_closed_; }

 private:
  std::unique_ptr<tsl::WritableFile> file_;
  int64_t position_ = 0;
  bool is_closed_ = false;
};

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_TSL_ARROW_OUTPUT_STREAM_H_
