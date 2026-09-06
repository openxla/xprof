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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "third_party/arrow/api.h"
#include "xprof/convert/events_db/schema.h"

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

TEST(ColumnTest, AccessorsAndMetadata) {
  Schema schema;
  const TypedFieldIndex<int64_t> field_idx =
      schema.RegisterFieldName<int64_t>("step");
  const Column col(field_idx, "step", 100);

  EXPECT_EQ(col.field_index(), field_idx);
  EXPECT_EQ(col.name(), "step");
  EXPECT_TRUE(
      col.ToArrowField()->Equals(*arrow::field("step", arrow::int64(), true)));
}

TEST(ColumnTest, ArithmeticColumn) {
  Column<int32_t> col(TypedFieldIndex<int32_t>{}, "int_col", 5);
  col.SetValue(0, 10);
  col.SetValue(1, 20);
  col.SetNull(2);
  col.SetValue(3, 40);
  col.SetValue(4, 50);
  col.SetNull(4);  // Override row 4 to null

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       col.ToArrowArray(5));

  const std::shared_ptr<arrow::Int32Array> int32_array =
      std::static_pointer_cast<arrow::Int32Array>(array);
  EXPECT_EQ(int32_array->length(), 5);
  EXPECT_EQ(int32_array->null_count(), 2);

  EXPECT_TRUE(int32_array->IsValid(0));
  EXPECT_EQ(int32_array->Value(0), 10);

  EXPECT_TRUE(int32_array->IsValid(1));
  EXPECT_EQ(int32_array->Value(1), 20);

  EXPECT_TRUE(int32_array->IsNull(2));

  EXPECT_TRUE(int32_array->IsValid(3));
  EXPECT_EQ(int32_array->Value(3), 40);

  EXPECT_TRUE(int32_array->IsNull(4));
}

TEST(ColumnTest, BooleanColumnMultiByte) {
  // Test across byte boundaries (18 rows spans 3 bytes: indices
  // 0..7, 8..15, 16..17)
  Column<bool> col(TypedFieldIndex<bool>{}, "bool_col", 18);
  col.SetValue(0, true);
  col.SetValue(1, false);
  col.SetNull(2);
  col.SetValue(7, true);
  col.SetValue(8, true);
  col.SetValue(9, false);
  col.SetValue(15, true);
  col.SetValue(16, false);
  col.SetValue(17, true);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       col.ToArrowArray(18));

  const std::shared_ptr<arrow::BooleanArray> bool_array =
      std::static_pointer_cast<arrow::BooleanArray>(array);
  EXPECT_EQ(bool_array->length(), 18);

  EXPECT_TRUE(bool_array->IsValid(0));
  EXPECT_TRUE(bool_array->Value(0));

  EXPECT_TRUE(bool_array->IsValid(1));
  EXPECT_FALSE(bool_array->Value(1));

  EXPECT_TRUE(bool_array->IsNull(2));

  EXPECT_TRUE(bool_array->IsValid(7));
  EXPECT_TRUE(bool_array->Value(7));

  EXPECT_TRUE(bool_array->IsValid(8));
  EXPECT_TRUE(bool_array->Value(8));

  EXPECT_TRUE(bool_array->IsValid(9));
  EXPECT_FALSE(bool_array->Value(9));

  EXPECT_TRUE(bool_array->IsValid(15));
  EXPECT_TRUE(bool_array->Value(15));

  EXPECT_TRUE(bool_array->IsValid(16));
  EXPECT_FALSE(bool_array->Value(16));

  EXPECT_TRUE(bool_array->IsValid(17));
  EXPECT_TRUE(bool_array->Value(17));
}

TEST(ColumnTest, StringColumn) {
  Column<std::string> col(TypedFieldIndex<std::string>{}, "str_col", 4);
  col.SetValue(0, "first");
  col.SetNull(1);
  col.SetValue(2, "third");
  col.SetValue(3, "");

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       col.ToArrowArray(4));

  const std::shared_ptr<arrow::StringArray> string_array =
      std::static_pointer_cast<arrow::StringArray>(array);
  EXPECT_EQ(string_array->length(), 4);
  EXPECT_EQ(string_array->null_count(), 1);

  EXPECT_TRUE(string_array->IsValid(0));
  EXPECT_EQ(string_array->GetString(0), "first");

  EXPECT_TRUE(string_array->IsNull(1));

  EXPECT_TRUE(string_array->IsValid(2));
  EXPECT_EQ(string_array->GetString(2), "third");

  EXPECT_TRUE(string_array->IsValid(3));
  EXPECT_EQ(string_array->GetString(3), "");
}

TEST(ColumnTest, VectorNumericColumn) {
  Column<std::vector<int32_t>> col(TypedFieldIndex<std::vector<int32_t>>{},
                                   "vec_col", 3);
  col.SetValue(0, {10, 20});
  col.SetNull(1);
  col.SetValue(2, {30, 40, 50});

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       col.ToArrowArray(3));

  const std::shared_ptr<arrow::ListArray> list_array =
      std::static_pointer_cast<arrow::ListArray>(array);
  EXPECT_EQ(list_array->length(), 3);
  EXPECT_EQ(list_array->null_count(), 1);

  EXPECT_TRUE(list_array->IsValid(0));
  EXPECT_EQ(list_array->value_length(0), 2);

  EXPECT_TRUE(list_array->IsNull(1));

  EXPECT_TRUE(list_array->IsValid(2));
  EXPECT_EQ(list_array->value_length(2), 3);

  const std::shared_ptr<arrow::Int32Array> values =
      std::static_pointer_cast<arrow::Int32Array>(list_array->values());
  EXPECT_EQ(values->length(), 5);
  EXPECT_EQ(values->Value(0), 10);
  EXPECT_EQ(values->Value(1), 20);
  EXPECT_EQ(values->Value(2), 30);
  EXPECT_EQ(values->Value(3), 40);
  EXPECT_EQ(values->Value(4), 50);
}

TEST(ColumnTest, VectorStringAndBoolColumn) {
  Column<std::vector<std::string>> str_vec_col(
      TypedFieldIndex<std::vector<std::string>>{}, "vec_str", 2);
  str_vec_col.SetValue(0, {"hello", "world"});
  str_vec_col.SetValue(1, {"parquet"});

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       str_vec_col.ToArrowArray(2));
  const std::shared_ptr<arrow::ListArray> string_array =
      std::static_pointer_cast<arrow::ListArray>(array);
  EXPECT_EQ(string_array->length(), 2);
  const std::shared_ptr<arrow::StringArray> string_values =
      std::static_pointer_cast<arrow::StringArray>(string_array->values());
  EXPECT_EQ(string_values->GetString(0), "hello");
  EXPECT_EQ(string_values->GetString(1), "world");
  EXPECT_EQ(string_values->GetString(2), "parquet");

  Column<std::vector<bool>> bool_vec_col(TypedFieldIndex<std::vector<bool>>{},
                                         "vec_bool", 2);
  bool_vec_col.SetValue(0, {true, false});
  bool_vec_col.SetValue(1, {true});

  ASSERT_OK_AND_ASSIGN(array, bool_vec_col.ToArrowArray(2));
  const std::shared_ptr<arrow::ListArray> bool_array =
      std::static_pointer_cast<arrow::ListArray>(array);
  EXPECT_EQ(bool_array->length(), 2);
  const std::shared_ptr<arrow::BooleanArray> bool_values =
      std::static_pointer_cast<arrow::BooleanArray>(bool_array->values());
  EXPECT_TRUE(bool_values->Value(0));
  EXPECT_FALSE(bool_values->Value(1));
  EXPECT_TRUE(bool_values->Value(2));
}

TEST(ColumnTest, BoundsAndEdgeCases) {
  Column<int64_t> col(TypedFieldIndex<int64_t>{}, "edge_col", 10);

  // Zero count
  const absl::StatusOr<std::shared_ptr<arrow::Array>> zero_array =
      col.ToArrowArray(0);
  ASSERT_THAT(zero_array, IsOk());
  EXPECT_EQ((*zero_array)->length(), 0);

  // Count exceeding capacity
  const absl::StatusOr<std::shared_ptr<arrow::Array>> exceed_capacity_array =
      col.ToArrowArray(11);
  EXPECT_THAT(exceed_capacity_array,
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Overflow count
  const absl::StatusOr<std::shared_ptr<arrow::Array>> overflow_array =
      col.ToArrowArray(std::numeric_limits<uint64_t>::max());
  EXPECT_THAT(overflow_array, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ColumnTest, ConcurrentDisjointWrites) {
  constexpr size_t kNumRows = 4000;
  constexpr size_t kNumThreads = 4;

  Column<int64_t> int_col(TypedFieldIndex<int64_t>{}, "concurrent_int",
                          kNumRows);
  Column<bool> bool_col(TypedFieldIndex<bool>{}, "concurrent_bool", kNumRows);

  {
    std::vector<std::jthread> threads;
    threads.reserve(kNumThreads);

    for (size_t t = 0; t < kNumThreads; ++t) {
      threads.emplace_back([&, t]() {
        for (size_t i = t; i < kNumRows; i += kNumThreads) {
          if (i % 3 == 0) {
            int_col.SetNull(i);
            bool_col.SetNull(i);
          } else {
            int_col.SetValue(i, static_cast<int64_t>(i * 10));
            bool_col.SetValue(i, (i % 2 == 0));
          }
        }
      });
    }
  }

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       int_col.ToArrowArray(kNumRows));
  const std::shared_ptr<arrow::Int64Array> int_array =
      std::static_pointer_cast<arrow::Int64Array>(array);
  EXPECT_EQ(int_array->length(), kNumRows);

  ASSERT_OK_AND_ASSIGN(array, bool_col.ToArrowArray(kNumRows));
  const std::shared_ptr<arrow::BooleanArray> bool_array =
      std::static_pointer_cast<arrow::BooleanArray>(array);
  EXPECT_EQ(bool_array->length(), kNumRows);

  for (size_t i = 0; i < kNumRows; ++i) {
    if (i % 3 == 0) {
      EXPECT_TRUE(int_array->IsNull(i));
      EXPECT_TRUE(bool_array->IsNull(i));
    } else {
      EXPECT_TRUE(int_array->IsValid(i));
      EXPECT_EQ(int_array->Value(i), static_cast<int64_t>(i * 10));

      EXPECT_TRUE(bool_array->IsValid(i));
      EXPECT_EQ(bool_array->Value(i), (i % 2 == 0));
    }
  }
}

TEST(ArrowUtilsTest, BytesForBits) {
  EXPECT_EQ(BytesForBits(0), 0);
  EXPECT_EQ(BytesForBits(1), 1);
  EXPECT_EQ(BytesForBits(7), 1);
  EXPECT_EQ(BytesForBits(8), 1);
  EXPECT_EQ(BytesForBits(9), 2);
  EXPECT_EQ(BytesForBits(16), 2);
  EXPECT_EQ(BytesForBits(17), 3);
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(BytesForBits(kMax), (kMax >> 3) + 1);
}

TEST(ColumnTest, MoveOnlySemantics) {
  static_assert(!std::is_copy_constructible_v<Column<int64_t>>);
  static_assert(!std::is_copy_assignable_v<Column<int64_t>>);
  static_assert(std::is_move_constructible_v<Column<int64_t>>);
  static_assert(std::is_move_assignable_v<Column<int64_t>>);

  Column<int64_t> col1(TypedFieldIndex<int64_t>{}, "move_col", 10);
  col1.SetValue(0, 42);

  Column<int64_t> col2 = std::move(col1);
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> array,
                       col2.ToArrowArray(1));
  const std::shared_ptr<arrow::Int64Array> int_array =
      std::static_pointer_cast<arrow::Int64Array>(array);
  EXPECT_EQ(int_array->Value(0), 42);
}

TEST(ColumnTest, SetFromRecord) {
  Schema schema;
  const TypedFieldIndex<int64_t> int_field =
      schema.RegisterFieldName<int64_t>("int_field");
  const TypedFieldIndex<std::string> str_field =
      schema.RegisterFieldName<std::string>("str_field");
  const TypedFieldIndex<bool> bool_field =
      schema.RegisterFieldName<bool>("bool_field");
  const TypedFieldIndex<std::vector<int32_t>> vec_field =
      schema.RegisterFieldName<std::vector<int32_t>>("vec_field");

  Column<int64_t> int_col(int_field, "int_field", 3);
  Column<std::string> str_col(str_field, "str_field", 3);
  Column<bool> bool_col(bool_field, "bool_field", 3);
  Column<std::vector<int32_t>> vec_col(vec_field, "vec_field", 3);

  // Row 0: All fields present.
  Record record0;
  record0[int_field] = 100;
  record0[str_field] = "hello";
  record0[bool_field] = true;
  record0[vec_field] = {1, 2, 3};

  int_col.Set(record0, 0);
  str_col.Set(record0, 0);
  bool_col.Set(record0, 0);
  vec_col.Set(record0, 0);

  // Row 1: Partial fields present (missing int_field and vec_field).
  Record record1;
  record1[str_field] = "world";
  record1[bool_field] = false;

  int_col.Set(record1, 1);
  str_col.Set(record1, 1);
  bool_col.Set(record1, 1);
  vec_col.Set(record1, 1);

  // Row 2: Empty record (all fields missing / null).
  Record record2;
  int_col.Set(record2, 2);
  str_col.Set(record2, 2);
  bool_col.Set(record2, 2);
  vec_col.Set(record2, 2);

  // Verify int_col.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> int_arr,
                       int_col.ToArrowArray(3));
  const std::shared_ptr<arrow::Int64Array> int_array =
      std::static_pointer_cast<arrow::Int64Array>(int_arr);
  EXPECT_EQ(int_array->length(), 3);
  EXPECT_TRUE(int_array->IsValid(0));
  EXPECT_EQ(int_array->Value(0), 100);
  EXPECT_TRUE(int_array->IsNull(1));
  EXPECT_TRUE(int_array->IsNull(2));

  // Verify str_col.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> str_arr,
                       str_col.ToArrowArray(3));
  const std::shared_ptr<arrow::StringArray> string_array =
      std::static_pointer_cast<arrow::StringArray>(str_arr);
  EXPECT_EQ(string_array->length(), 3);
  EXPECT_TRUE(string_array->IsValid(0));
  EXPECT_EQ(string_array->GetString(0), "hello");
  EXPECT_TRUE(string_array->IsValid(1));
  EXPECT_EQ(string_array->GetString(1), "world");
  EXPECT_TRUE(string_array->IsNull(2));

  // Verify bool_col.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> bool_arr,
                       bool_col.ToArrowArray(3));
  const std::shared_ptr<arrow::BooleanArray> bool_array =
      std::static_pointer_cast<arrow::BooleanArray>(bool_arr);
  EXPECT_EQ(bool_array->length(), 3);
  EXPECT_TRUE(bool_array->IsValid(0));
  EXPECT_TRUE(bool_array->Value(0));
  EXPECT_TRUE(bool_array->IsValid(1));
  EXPECT_FALSE(bool_array->Value(1));
  EXPECT_TRUE(bool_array->IsNull(2));

  // Verify vec_col.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Array> vec_arr,
                       vec_col.ToArrowArray(3));
  const std::shared_ptr<arrow::ListArray> list_array =
      std::static_pointer_cast<arrow::ListArray>(vec_arr);
  EXPECT_EQ(list_array->length(), 3);
  EXPECT_TRUE(list_array->IsValid(0));
  EXPECT_EQ(list_array->value_length(0), 3);
  EXPECT_TRUE(list_array->IsNull(1));
  EXPECT_TRUE(list_array->IsNull(2));
}

}  // namespace
}  // namespace xprof::events_db::internal
