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

#include "xprof/convert/events_db/tsl_arrow_output_stream.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/arrow/result.h"
#include "third_party/arrow/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xprof/convert/events_db/arrow_utils.h"

namespace xprof::events_db::internal {

TslArrowOutputStream::TslArrowOutputStream(
    absl_nonnull std::unique_ptr<tsl::WritableFile> file)
    : file_(std::move(file)) {
  // `arrow::io::FileMode` is defined in Arrow's internal `type_fwd.h` and
  // provided transitively via `interfaces.h`; suppress `misc-include-cleaner`
  // since `type_fwd.h` is not meant to be included directly.
  set_mode(arrow::io::FileMode::WRITE);  // NOLINT(misc-include-cleaner)
}

absl::StatusOr<std::shared_ptr<TslArrowOutputStream>>
TslArrowOutputStream::Open(absl::string_view file_path, tsl::Env* env) {
  if (env == nullptr) env = tsl::Env::Default();
  std::unique_ptr<tsl::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(std::string(file_path), &file));
  return std::make_shared<TslArrowOutputStream>(std::move(file));
}

TslArrowOutputStream::~TslArrowOutputStream() {
  if (is_closed_ || file_ == nullptr) return;
  static_cast<void>(file_->Close());
}

arrow::Status TslArrowOutputStream::Write(const void* data, int64_t nbytes) {
  if (is_closed_) return arrow::Status::Invalid("Write on a closed stream.");
  if (nbytes < 0) return arrow::Status::Invalid("Write with negative nbytes.");
  if (nbytes == 0) return arrow::Status::OK();
  if (data == nullptr) {
    return arrow::Status::Invalid("Write with null data pointer.");
  }
  RETURN_NOT_OK(internal::ToArrowStatus(file_->Append(
      absl::string_view(static_cast<const char*>(data), nbytes))));
  position_ += nbytes;
  return arrow::Status::OK();
}

arrow::Status TslArrowOutputStream::Flush() {
  if (is_closed_) return arrow::Status::Invalid("Flush on a closed stream.");
  return internal::ToArrowStatus(file_->Flush());
}

arrow::Status TslArrowOutputStream::Close() {
  if (is_closed_) return arrow::Status::OK();
  is_closed_ = true;
  if (file_ == nullptr) return arrow::Status::OK();
  return internal::ToArrowStatus(file_->Close());
}

arrow::Result<int64_t> TslArrowOutputStream::Tell() const {
  if (is_closed_) return arrow::Status::Invalid("Tell on a closed stream.");
  return position_;
}

}  // namespace xprof::events_db::internal
