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

#include "xprof/convert/events_db/arrow_utils.h"

#include "absl/status/status.h"
#include "arrow/api.h"  // from @arrow

namespace xprof::events_db::internal {

absl::Status ToAbslStatus(const arrow::Status& status) {
  if (status.ok()) return absl::OkStatus();
  switch (status.code()) {
    case arrow::StatusCode::OutOfMemory:
    case arrow::StatusCode::CapacityError:
      return absl::ResourceExhaustedError(status.message());
    case arrow::StatusCode::KeyError:
      return absl::NotFoundError(status.message());
    case arrow::StatusCode::TypeError:
    case arrow::StatusCode::Invalid:
    case arrow::StatusCode::ExpressionValidationError:
      return absl::InvalidArgumentError(status.message());
    case arrow::StatusCode::IOError:
      return absl::UnavailableError(status.message());
    case arrow::StatusCode::IndexError:
      return absl::OutOfRangeError(status.message());
    case arrow::StatusCode::Cancelled:
      return absl::CancelledError(status.message());
    case arrow::StatusCode::NotImplemented:
      return absl::UnimplementedError(status.message());
    case arrow::StatusCode::SerializationError:
      return absl::DataLossError(status.message());
    case arrow::StatusCode::AlreadyExists:
      return absl::AlreadyExistsError(status.message());
    case arrow::StatusCode::CodeGenError:
    case arrow::StatusCode::ExecutionError:
      return absl::InternalError(status.message());
    default:
      return absl::UnknownError(status.message());
  }
}

arrow::Status ToArrowStatus(const absl::Status& status) {
  if (status.ok()) return arrow::Status::OK();
  switch (status.code()) {
    case absl::StatusCode::kResourceExhausted:
      return arrow::Status::CapacityError(status.message());
    case absl::StatusCode::kNotFound:
      return arrow::Status::KeyError(status.message());
    case absl::StatusCode::kInvalidArgument:
    case absl::StatusCode::kFailedPrecondition:
      return arrow::Status::Invalid(status.message());
    case absl::StatusCode::kOutOfRange:
      return arrow::Status::IndexError(status.message());
    case absl::StatusCode::kCancelled:
    case absl::StatusCode::kDeadlineExceeded:
    case absl::StatusCode::kAborted:
      return arrow::Status::Cancelled(status.message());
    case absl::StatusCode::kUnimplemented:
      return arrow::Status::NotImplemented(status.message());
    case absl::StatusCode::kDataLoss:
      return arrow::Status::SerializationError(status.message());
    case absl::StatusCode::kAlreadyExists:
      return arrow::Status::AlreadyExists(status.message());
    case absl::StatusCode::kUnavailable:
      return arrow::Status::IOError(status.message());
    case absl::StatusCode::kInternal:
      return arrow::Status::ExecutionError(status.message());
    default:
      return arrow::Status::UnknownError(status.message());
  }
}

}  // namespace xprof::events_db::internal
