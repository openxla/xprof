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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_PARQUET_RECORD_CONSUMER_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_PARQUET_RECORD_CONSUMER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "arrow/util/type_fwd.h"  // from @arrow
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/executor.h"
#include "xprof/convert/executor_factory.h"

namespace xprof::events_db {

// Returns a single-threaded executor
// (`CreateXprofThreadPoolExecutor("parquet_writer", 1)`) for asynchronous
// background batch writing.
std::unique_ptr<tensorflow::profiler::Executor>
DefaultParquetWriterExecutorFactory();

// Configuration options for exporting events DB records to Apache Parquet.
struct ParquetExportOptions {
  // If set, at most this many records will be written before early stopping.
  std::optional<uint64_t> max_record_count = std::nullopt;

  // Number of records buffered before flushing a batch to disk.
  uint32_t batch_size = 65536;

  // Compression codec applied to Parquet data pages (e.g.
  // `arrow::Compression::UNCOMPRESSED`, `arrow::Compression::SNAPPY`,
  // `arrow::Compression::ZSTD`). If unset (`std::nullopt`), no compression
  // codec is explicitly configured on the underlying writer.
  std::optional<arrow::Compression::type> compression_type = std::nullopt;

  // Compressor-specific compression level (e.g. 1-22 for ZSTD, 1-9 for GZIP).
  // If unset (`std::nullopt`), no compression level is explicitly configured
  // on the underlying writer, leaving it to the compressor's default behavior.
  std::optional<int> compression_level = std::nullopt;
};

// Thread-safe consumer that streams and writes `Record` instances to an Apache
// Parquet file in batches.
class ParquetRecordConsumer {
 public:
  // Creates a `ParquetRecordConsumer` with a custom executor factory.
  //
  // Note: The consumer enqueues at most one background task at a time (to flush
  // completed batches sequentially). Supplying an `executor_factory` that
  // creates a multi-threaded executor does not increase parallelism and wastes
  // resources; a single-threaded executor is strongly recommended (such as
  // `DefaultParquetWriterExecutorFactory`).
  static absl::StatusOr<ParquetRecordConsumer> Build(
      Schema& schema, absl::string_view file_path,
      tensorflow::profiler::ExecutorFactoryRef executor_factory,
      ParquetExportOptions options = {});

  // Creates a `ParquetRecordConsumer` using the default single-threaded
  // executor factory (`DefaultParquetWriterExecutorFactory`).
  static absl::StatusOr<ParquetRecordConsumer> Build(
      Schema& schema, absl::string_view file_path,
      ParquetExportOptions options = {}) {
    return Build(schema, file_path, DefaultParquetWriterExecutorFactory,
                 std::move(options));
  }

  ~ParquetRecordConsumer();

  ParquetRecordConsumer(const ParquetRecordConsumer&) = delete;
  ParquetRecordConsumer(ParquetRecordConsumer&&);

  ParquetRecordConsumer& operator=(const ParquetRecordConsumer&) = delete;
  ParquetRecordConsumer& operator=(ParquetRecordConsumer&&);

  // Appends a record to the in-memory batch. Flushes to Parquet when
  // `batch_size` is reached. Returns `StepControl::kStop` if `max_record_count`
  // is specified and reached. If `max_record_count` is not specified, returns
  // failure if called more than 2^64 times.
  absl::StatusOr<StepControl> Consume(Record& record);

  // Flushes any remaining buffered records and closes the Parquet file.
  absl::Status Finalize(
      const absl::StatusOr<ParseStatus>& result = ParseStatus::kComplete);

 private:
  struct Impl;

  explicit ParquetRecordConsumer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace xprof::events_db

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_PARQUET_RECORD_CONSUMER_H_
