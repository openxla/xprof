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

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "third_party/arrow/api.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(ArrowUtilsTest, ToStatusSuccess) {
  EXPECT_THAT(ToAbslStatus(arrow::Status::OK()), IsOk());
}

TEST(ArrowUtilsTest, ToStatusErrorMappings) {
  EXPECT_THAT(ToAbslStatus(arrow::Status::OutOfMemory("oom message")),
              StatusIs(absl::StatusCode::kResourceExhausted, "oom message"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::CapacityError("cap message")),
              StatusIs(absl::StatusCode::kResourceExhausted, "cap message"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::KeyError("key not found")),
              StatusIs(absl::StatusCode::kNotFound, "key not found"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::TypeError("type error")),
              StatusIs(absl::StatusCode::kInvalidArgument, "type error"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::Invalid("invalid argument")),
              StatusIs(absl::StatusCode::kInvalidArgument, "invalid argument"));
  EXPECT_THAT(
      ToAbslStatus(arrow::Status::ExpressionValidationError("expr invalid")),
      StatusIs(absl::StatusCode::kInvalidArgument, "expr invalid"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::IOError("io failure")),
              StatusIs(absl::StatusCode::kUnavailable, "io failure"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::IndexError("index out of bounds")),
              StatusIs(absl::StatusCode::kOutOfRange, "index out of bounds"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::Cancelled("cancelled op")),
              StatusIs(absl::StatusCode::kCancelled, "cancelled op"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::NotImplemented("not implemented")),
              StatusIs(absl::StatusCode::kUnimplemented, "not implemented"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::SerializationError("ser error")),
              StatusIs(absl::StatusCode::kDataLoss, "ser error"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::AlreadyExists("file exists")),
              StatusIs(absl::StatusCode::kAlreadyExists, "file exists"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::CodeGenError("codegen fail")),
              StatusIs(absl::StatusCode::kInternal, "codegen fail"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::ExecutionError("exec fail")),
              StatusIs(absl::StatusCode::kInternal, "exec fail"));
  EXPECT_THAT(ToAbslStatus(arrow::Status::UnknownError("unknown error")),
              StatusIs(absl::StatusCode::kUnknown, "unknown error"));
}

TEST(ArrowUtilsTest, ToArrowStatusSuccess) {
  EXPECT_EQ(ToArrowStatus(absl::OkStatus()), arrow::Status::OK());
}

TEST(ArrowUtilsTest, ToArrowStatusErrorMappings) {
  EXPECT_EQ(ToArrowStatus(absl::ResourceExhaustedError("oom")),
            arrow::Status::CapacityError("oom"));
  EXPECT_EQ(ToArrowStatus(absl::NotFoundError("not found")),
            arrow::Status::KeyError("not found"));
  EXPECT_EQ(ToArrowStatus(absl::InvalidArgumentError("invalid")),
            arrow::Status::Invalid("invalid"));
  EXPECT_EQ(ToArrowStatus(absl::FailedPreconditionError("precondition")),
            arrow::Status::Invalid("precondition"));
  EXPECT_EQ(ToArrowStatus(absl::OutOfRangeError("out of range")),
            arrow::Status::IndexError("out of range"));
  EXPECT_EQ(ToArrowStatus(absl::CancelledError("cancelled")),
            arrow::Status::Cancelled("cancelled"));
  EXPECT_EQ(ToArrowStatus(absl::DeadlineExceededError("deadline")),
            arrow::Status::Cancelled("deadline"));
  EXPECT_EQ(ToArrowStatus(absl::AbortedError("aborted")),
            arrow::Status::Cancelled("aborted"));
  EXPECT_EQ(ToArrowStatus(absl::UnimplementedError("unimplemented")),
            arrow::Status::NotImplemented("unimplemented"));
  EXPECT_EQ(ToArrowStatus(absl::DataLossError("data loss")),
            arrow::Status::SerializationError("data loss"));
  EXPECT_EQ(ToArrowStatus(absl::AlreadyExistsError("exists")),
            arrow::Status::AlreadyExists("exists"));
  EXPECT_EQ(ToArrowStatus(absl::UnavailableError("unavailable")),
            arrow::Status::IOError("unavailable"));
  EXPECT_EQ(ToArrowStatus(absl::InternalError("internal")),
            arrow::Status::ExecutionError("internal"));
  EXPECT_EQ(ToArrowStatus(absl::UnknownError("unknown")),
            arrow::Status::UnknownError("unknown"));
}

TEST(ArrowUtilsTest, ToAbslStatusOr) {
  const arrow::Result<int> ok_result(42);
  EXPECT_THAT(ToAbslStatusOr(ok_result), IsOkAndHolds(42));

  arrow::Result<std::string> str_result("arrow string");
  EXPECT_THAT(ToAbslStatusOr(std::move(str_result)),
              IsOkAndHolds("arrow string"));

  const arrow::Result<int> error_result(
      arrow::Status::KeyError("key not found"));
  EXPECT_THAT(ToAbslStatusOr(error_result),
              StatusIs(absl::StatusCode::kNotFound, "key not found"));
}

TEST(ArrowUtilsTest, ToArrowResult) {
  const absl::StatusOr<int> ok_status(42);
  const arrow::Result<int> arrow_ok = ToArrowResult(ok_status);
  ASSERT_TRUE(arrow_ok.ok());
  EXPECT_EQ(*arrow_ok, 42);

  absl::StatusOr<std::string> str_status("absl string");
  const arrow::Result<std::string> arrow_str =
      ToArrowResult(std::move(str_status));
  ASSERT_TRUE(arrow_str.ok());
  EXPECT_EQ(*arrow_str, "absl string");

  const absl::StatusOr<int> error_status = absl::NotFoundError("missing key");
  const arrow::Result<int> arrow_err = ToArrowResult(error_status);
  ASSERT_FALSE(arrow_err.ok());
  EXPECT_EQ(arrow_err.status().code(), arrow::StatusCode::KeyError);
  EXPECT_EQ(arrow_err.status().message(), "missing key");
}

TEST(ArrowUtilsTest, IsStdVectorTrait) {
  static_assert(is_std_vector_v<std::vector<int>>);
  static_assert(is_std_vector_v<std::vector<std::string>>);
  static_assert(is_std_vector_v<std::vector<std::vector<double>>>);

  static_assert(!is_std_vector_v<int>);
  static_assert(!is_std_vector_v<std::string>);
  static_assert(!is_std_vector_v<std::monostate>);
  static_assert(!is_std_vector_v<bool>);
}

TEST(ArrowUtilsTest, GetArrowType) {
  EXPECT_TRUE(GetArrowType<std::monostate>()->Equals(*arrow::null()));
  EXPECT_TRUE(GetArrowType<bool>()->Equals(*arrow::boolean()));
  EXPECT_TRUE(GetArrowType<int32_t>()->Equals(*arrow::int32()));
  EXPECT_TRUE(GetArrowType<uint32_t>()->Equals(*arrow::uint32()));
  EXPECT_TRUE(GetArrowType<int64_t>()->Equals(*arrow::int64()));
  EXPECT_TRUE(GetArrowType<uint64_t>()->Equals(*arrow::uint64()));
  EXPECT_TRUE(GetArrowType<double>()->Equals(*arrow::float64()));
  EXPECT_TRUE(GetArrowType<std::string>()->Equals(*arrow::utf8()));

  EXPECT_TRUE(GetArrowType<std::vector<int64_t>>()->Equals(
      *arrow::list(arrow::int64())));
  EXPECT_TRUE(GetArrowType<std::vector<std::string>>()->Equals(
      *arrow::list(arrow::utf8())));
}

}  // namespace
}  // namespace xprof::events_db::internal
