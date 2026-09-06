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

#include "xprof/convert/events_db/parquet_record_consumer.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/arrow/api.h"
#include "third_party/arrow/io/api.h"  // IWYU pragma: keep
#include "third_party/arrow/util/type_fwd.h"
#include "third_party/parquet_cpp/src2/parquet/arrow/reader.h"
#include "xprof/convert/events_db/arrow_utils.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/executor.h"

namespace xprof::events_db {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::xprof::events_db::internal::FieldIndices;

absl::StatusOr<std::shared_ptr<arrow::Table>> ReadParquetFile(
    absl::string_view file_path) {
  // `arrow::io::ReadableFile` is defined in `arrow/io/file.h` and provided
  // transitively via `arrow/io/api.h`; suppress `misc-include-cleaner` since
  // `file.h` is not exported for direct inclusion.
  ASSIGN_OR_RETURN(
      std::shared_ptr<arrow::io::ReadableFile>  // NOLINT(misc-include-cleaner)
          infile_res,
      internal::ToAbslStatusOr(
          arrow::io::ReadableFile::Open(  // NOLINT(misc-include-cleaner)
              std::string(file_path))));
  ASSIGN_OR_RETURN(std::unique_ptr<parquet::arrow::FileReader> reader,
                   internal::ToAbslStatusOr(parquet::arrow::OpenFile(
                       infile_res, arrow::default_memory_pool())));
  std::shared_ptr<arrow::Table> table;
  RETURN_IF_ERROR(internal::ToAbslStatus(reader->ReadTable(&table)));
  return table;
}

TEST(ParquetRecordConsumerTest, InvalidBatchSizeReturnsError) {
  Schema schema;
  ParquetExportOptions options;
  options.batch_size = 0;
  EXPECT_THAT(ParquetRecordConsumer::Build(schema, "/tmp/out.parquet", options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ParquetRecordConsumerTest, InvalidPathReturnsError) {
  Schema schema;
  EXPECT_THAT(ParquetRecordConsumer::Build(
                  schema, "/nonexistent_dir_12345/sub/test.parquet"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ParquetRecordConsumerTest, EmptyConsumerCreatesValidParquet) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/empty_test.parquet");
  Schema schema;
  ASSERT_OK_AND_ASSIGN(ParquetRecordConsumer consumer,
                       ParquetRecordConsumer::Build(schema, file_path));

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 31);
}

TEST(ParquetRecordConsumerTest, WriteAndReadBackSingleBatch) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/single_batch_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ASSERT_OK_AND_ASSIGN(ParquetRecordConsumer consumer,
                       ParquetRecordConsumer::Build(schema, file_path));

  // Row 0: Full record
  Record record0;
  record0[indices.kernel_name] = "matmul_kernel";
  record0[indices.step] = "step_42";
  record0[indices.start_ns] = 1000ULL;
  record0[indices.end_ns] = 2000ULL;
  record0[indices.input_tensors] = {"tensor_a", "tensor_b"};
  EXPECT_THAT(consumer.Consume(record0),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  // Row 1: Partial record (missing step and input_tensors)
  Record record1;
  record1[indices.kernel_name] = "add_kernel";
  record1[indices.start_ns] = 3000ULL;
  record1[indices.end_ns] = 3500ULL;
  EXPECT_THAT(consumer.Consume(record1),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), 2);

  // Validate kernel_name column
  const std::shared_ptr<arrow::ChunkedArray> kernel_col =
      table->GetColumnByName("kernel_name");
  ASSERT_NE(kernel_col, nullptr);
  const std::shared_ptr<arrow::StringArray> kernel_array =
      std::static_pointer_cast<arrow::StringArray>(kernel_col->chunk(0));
  EXPECT_EQ(kernel_array->GetString(0), "matmul_kernel");
  EXPECT_EQ(kernel_array->GetString(1), "add_kernel");

  // Validate step column
  const std::shared_ptr<arrow::ChunkedArray> step_col =
      table->GetColumnByName("step");
  ASSERT_NE(step_col, nullptr);
  const std::shared_ptr<arrow::StringArray> step_array =
      std::static_pointer_cast<arrow::StringArray>(step_col->chunk(0));
  EXPECT_TRUE(step_array->IsValid(0));
  EXPECT_EQ(step_array->GetString(0), "step_42");
  EXPECT_TRUE(step_array->IsNull(1));

  // Validate start_ns column
  const std::shared_ptr<arrow::ChunkedArray> start_col =
      table->GetColumnByName("start_ns");
  ASSERT_NE(start_col, nullptr);
  const std::shared_ptr<arrow::UInt64Array> start_array =
      std::static_pointer_cast<arrow::UInt64Array>(start_col->chunk(0));
  EXPECT_EQ(start_array->Value(0), 1000ULL);
  EXPECT_EQ(start_array->Value(1), 3000ULL);

  // Validate input_tensors list column
  const std::shared_ptr<arrow::ChunkedArray> input_tensors_col =
      table->GetColumnByName("input_tensors");
  ASSERT_NE(input_tensors_col, nullptr);
  const std::shared_ptr<arrow::ListArray> list_array =
      std::static_pointer_cast<arrow::ListArray>(input_tensors_col->chunk(0));
  EXPECT_TRUE(list_array->IsValid(0));
  EXPECT_EQ(list_array->value_length(0), 2);
  EXPECT_TRUE(list_array->IsNull(1));
}

TEST(ParquetRecordConsumerTest, MultiBatchBufferingAndFlushing) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/multi_batch_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ParquetExportOptions options;
  options.batch_size = 2;  // Small batch size to trigger multiple flushes

  ASSERT_OK_AND_ASSIGN(
      ParquetRecordConsumer consumer,
      ParquetRecordConsumer::Build(schema, file_path, options));

  constexpr int kNumRecords = 5;
  for (int i = 0; i < kNumRecords; ++i) {
    Record record;
    record[indices.kernel_name] = absl::StrCat("kernel_", i);
    record[indices.start_ns] = static_cast<uint64_t>(i * 100);
    EXPECT_THAT(consumer.Consume(record),
                IsOkAndHolds(Eq(StepControl::kContinue)));
  }

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), kNumRecords);

  const std::shared_ptr<arrow::ChunkedArray> kernel_col =
      table->GetColumnByName("kernel_name");
  ASSERT_NE(kernel_col, nullptr);

  int current_idx = 0;
  for (int c = 0; c < kernel_col->num_chunks(); ++c) {
    const std::shared_ptr<arrow::StringArray> chunk =
        std::static_pointer_cast<arrow::StringArray>(kernel_col->chunk(c));
    for (int64_t r = 0; r < chunk->length(); ++r) {
      EXPECT_EQ(chunk->GetString(r), absl::StrCat("kernel_", current_idx++));
    }
  }
}

TEST(ParquetRecordConsumerTest, MaxRecordCountEarlyStopping) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/max_record_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ParquetExportOptions options;
  options.batch_size = 2;
  options.max_record_count = 3;

  ASSERT_OK_AND_ASSIGN(
      ParquetRecordConsumer consumer,
      ParquetRecordConsumer::Build(schema, file_path, options));

  Record record0;
  record0[indices.kernel_name] = "k0";
  EXPECT_THAT(consumer.Consume(record0),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  Record record1;
  record1[indices.kernel_name] = "k1";
  EXPECT_THAT(consumer.Consume(record1),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  Record record2;
  record2[indices.kernel_name] = "k2";
  EXPECT_THAT(consumer.Consume(record2),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  // 4th record exceeds max_record_count (3)
  Record record3;
  record3[indices.kernel_name] = "k3";
  EXPECT_THAT(consumer.Consume(record3), IsOkAndHolds(Eq(StepControl::kStop)));

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), 3);
}

TEST(ParquetRecordConsumerTest, CompressionOptionsConfigured) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/compressed_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ParquetExportOptions options;
  options.batch_size = 2;
  options.compression_type = arrow::Compression::ZSTD;
  options.compression_level = 3;

  ASSERT_OK_AND_ASSIGN(
      ParquetRecordConsumer consumer,
      ParquetRecordConsumer::Build(schema, file_path, options));

  Record record;
  record[indices.kernel_name] = "compressed_kernel";
  EXPECT_THAT(consumer.Consume(record),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), 1);
  const std::shared_ptr<arrow::ChunkedArray> kernel_col =
      table->GetColumnByName("kernel_name");
  ASSERT_NE(kernel_col, nullptr);
  const std::shared_ptr<arrow::StringArray> kernel_array =
      std::static_pointer_cast<arrow::StringArray>(kernel_col->chunk(0));
  EXPECT_EQ(kernel_array->GetString(0), "compressed_kernel");
}

TEST(ParquetRecordConsumerTest, CompressionTypeWithoutLevel) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/compressed_no_level_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ParquetExportOptions options;
  options.batch_size = 2;
  options.compression_type = arrow::Compression::ZSTD;
  // options.compression_level left as std::nullopt

  ASSERT_OK_AND_ASSIGN(
      ParquetRecordConsumer consumer,
      ParquetRecordConsumer::Build(schema, file_path, options));

  Record record;
  record[indices.kernel_name] = "compressed_kernel";
  EXPECT_THAT(consumer.Consume(record),
              IsOkAndHolds(Eq(StepControl::kContinue)));

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), 1);
}

TEST(ParquetRecordConsumerTest, CompressionLevelRequiresType) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/compression_level_no_type.parquet");
  Schema schema;
  ParquetExportOptions options;
  options.compression_level = 3;

  EXPECT_THAT(
      ParquetRecordConsumer::Build(schema, file_path, options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "compression_level requires compression_type to be set."));
}

TEST(ParquetRecordConsumerTest, DestructorJoinsExecutorWithoutFinalize) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/dtor_join_test.parquet");
  Schema schema;
  std::atomic<bool> joined{false};

  class SpyExecutor : public tensorflow::profiler::Executor {
   public:
    SpyExecutor(std::atomic<bool>& joined) : joined_(joined) {}
    void Execute(std::function<void()> fn) override { fn(); }
    void JoinAll() override { joined_ = true; }
    std::atomic<bool>& joined_;
  };

  const auto factory =
      [&joined]() -> std::unique_ptr<tensorflow::profiler::Executor> {
    return std::make_unique<SpyExecutor>(joined);
  };

  {
    ASSERT_OK_AND_ASSIGN(
        ParquetRecordConsumer consumer,
        ParquetRecordConsumer::Build(schema, file_path, factory));
    EXPECT_FALSE(joined.load());
  }
  EXPECT_TRUE(joined.load());
}

TEST(ParquetRecordConsumerTest, FinalizeWithErrorPropagatesStatus) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/error_finalize_test.parquet");
  Schema schema;
  ASSERT_OK_AND_ASSIGN(ParquetRecordConsumer consumer,
                       ParquetRecordConsumer::Build(schema, file_path));

  EXPECT_THAT(consumer.Finalize(absl::InternalError("parser failed")),
              StatusIs(absl::StatusCode::kInternal, "parser failed"));
}

TEST(ParquetRecordConsumerTest, MoveConstructible) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/move_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ASSERT_OK_AND_ASSIGN(ParquetRecordConsumer consumer1,
                       ParquetRecordConsumer::Build(schema, file_path));
  ParquetRecordConsumer consumer2 = std::move(consumer1);

  Record record;
  record[indices.kernel_name] = "moved_kernel";
  EXPECT_THAT(consumer2.Consume(record),
              IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_OK(consumer2.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  EXPECT_EQ(table->num_rows(), 1);
}

TEST(ParquetRecordConsumerTest, MoveAssignable) {
  const std::string file_path1 =
      absl::StrCat(testing::TempDir(), "/move_assign_test1.parquet");
  const std::string file_path2 =
      absl::StrCat(testing::TempDir(), "/move_assign_test2.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ASSERT_OK_AND_ASSIGN(ParquetRecordConsumer consumer1,
                       ParquetRecordConsumer::Build(schema, file_path1));
  ASSERT_OK_AND_ASSIGN(ParquetRecordConsumer consumer2,
                       ParquetRecordConsumer::Build(schema, file_path2));

  consumer2 = std::move(consumer1);

  Record record;
  record[indices.kernel_name] = "move_assigned_kernel";
  EXPECT_THAT(consumer2.Consume(record),
              IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_OK(consumer2.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path1));
  EXPECT_EQ(table->num_rows(), 1);
}

TEST(ParquetRecordConsumerTest, ConcurrentConsume) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/concurrent_consume_test.parquet");
  Schema schema;
  const FieldIndices indices(schema);

  ParquetExportOptions options;
  options.batch_size = 64;

  ASSERT_OK_AND_ASSIGN(
      ParquetRecordConsumer consumer,
      ParquetRecordConsumer::Build(schema, file_path, options));

  constexpr int kNumThreads = 8;
  constexpr int kRecordsPerThread = 200;
  constexpr int kTotalRecords = kNumThreads * kRecordsPerThread;

  std::vector<absl::StatusOr<StepControl>> thread_statuses(
      kNumThreads, StepControl::kContinue);
  {
    std::vector<std::jthread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      threads.emplace_back([&, t]() {
        for (int i = 0; i < kRecordsPerThread; ++i) {
          Record record;
          record[indices.kernel_name] = absl::StrCat("kernel_", t, "_", i);
          record[indices.thread_id] = static_cast<uint64_t>(t);
          record[indices.start_ns] = static_cast<uint64_t>(t * 1000000 + i);
          thread_statuses[t] = consumer.Consume(record);
          if (!thread_statuses[t].ok()) break;
        }
      });
    }
  }

  for (int t = 0; t < kNumThreads; ++t) {
    EXPECT_THAT(thread_statuses[t], IsOkAndHolds(Eq(StepControl::kContinue)));
  }

  EXPECT_OK(consumer.Finalize());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::Table> table,
                       ReadParquetFile(file_path));
  ASSERT_EQ(table->num_rows(), kTotalRecords);

  const std::shared_ptr<arrow::ChunkedArray> kernel_col =
      table->GetColumnByName("kernel_name");
  ASSERT_NE(kernel_col, nullptr);

  absl::flat_hash_set<std::string> seen_kernels;
  seen_kernels.reserve(kTotalRecords);
  for (int c = 0; c < kernel_col->num_chunks(); ++c) {
    const std::shared_ptr<arrow::StringArray> chunk =
        std::static_pointer_cast<arrow::StringArray>(kernel_col->chunk(c));
    for (int64_t r = 0; r < chunk->length(); ++r) {
      seen_kernels.insert(chunk->GetString(r));
    }
  }

  absl::flat_hash_set<std::string> expected_kernels;
  expected_kernels.reserve(kTotalRecords);
  for (int t = 0; t < kNumThreads; ++t) {
    for (int i = 0; i < kRecordsPerThread; ++i) {
      expected_kernels.insert(absl::StrCat("kernel_", t, "_", i));
    }
  }

  EXPECT_EQ(seen_kernels, expected_kernels);
}

}  // namespace
}  // namespace xprof::events_db
