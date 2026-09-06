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

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/arrow/api.h"
#include "third_party/arrow/io/api.h"  // IWYU pragma: keep
#include "third_party/parquet_cpp/src2/parquet/arrow/writer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/events_db/arrow_utils.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/tsl_arrow_output_stream.h"
#include "xprof/convert/executor.h"
#include "xprof/convert/executor_factory.h"

namespace xprof::events_db {
namespace {

auto CreateColumns(const internal::FieldIndices& indices, uint32_t batch_size) {
#define MAKE_COLUMN(name) internal::Column(indices.name, #name, batch_size)
  return std::make_tuple(
      MAKE_COLUMN(device), MAKE_COLUMN(stream_id), MAKE_COLUMN(thread_id),
      MAKE_COLUMN(thread_name), MAKE_COLUMN(correlation_id),
      MAKE_COLUMN(kernel_name), MAKE_COLUMN(kernel_details),
      MAKE_COLUMN(tf_op_name), MAKE_COLUMN(tf_op_type), MAKE_COLUMN(hlo_op),
      MAKE_COLUMN(hlo_module), MAKE_COLUMN(start_ns), MAKE_COLUMN(end_ns),
      MAKE_COLUMN(self_time_ns), MAKE_COLUMN(step), MAKE_COLUMN(category),
      MAKE_COLUMN(flops), MAKE_COLUMN(memory_accessed),
      MAKE_COLUMN(input_tensors), MAKE_COLUMN(output_tensors),
      MAKE_COLUMN(trace_args), MAKE_COLUMN(hlo_fingerprint), MAKE_COLUMN(flow),
      MAKE_COLUMN(source_line), MAKE_COLUMN(dcn_src_slice_id),
      MAKE_COLUMN(dcn_dst_slice_id), MAKE_COLUMN(dcn_src_logical_device_id),
      MAKE_COLUMN(dcn_dst_logical_device_id), MAKE_COLUMN(dcn_collective_name),
      MAKE_COLUMN(dcn_duration_us), MAKE_COLUMN(dcn_payload_size_bytes));
#undef MAKE_COLUMN
}

using ColumnsType =
    decltype(CreateColumns(std::declval<internal::FieldIndices>(), 0));

std::shared_ptr<arrow::Schema> CreateArrowSchema(const ColumnsType& columns) {
  arrow::FieldVector fields(std::apply(
      [&](auto&... cols) { return arrow::FieldVector{cols.ToArrowField()...}; },
      columns));
  return arrow::schema(std::move(fields));
}

struct Batch {
  Batch(const internal::FieldIndices& indices, uint32_t batch_size)
      : columns(CreateColumns(indices, batch_size)) {}

  void Fill(Record& record, uint32_t index) {
    std::apply([&](auto&... cols) { (cols.Set(record, index), ...); }, columns);
  }

  absl::StatusOr<std::vector<std::shared_ptr<arrow::Array>>> ToArrowArrays(
      uint32_t count) const {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(std::tuple_size_v<ColumnsType>);
    absl::Status status = absl::OkStatus();
    auto append = [&](const auto& column) {
      if (!status.ok()) return;
      absl::StatusOr<std::shared_ptr<arrow::Array>> array =
          column.ToArrowArray(count);
      if (array.ok())
        arrays.push_back(std::move(*array));
      else
        status = array.status();
    };
    std::apply([&](const auto&... cols) { (append(cols), ...); }, columns);
    TF_RETURN_IF_ERROR(status);
    return arrays;
  }

  ColumnsType columns;
  std::atomic<uint32_t> ready_count{0};
};

}  // namespace

std::unique_ptr<tensorflow::profiler::Executor>
DefaultParquetWriterExecutorFactory() {
  return tensorflow::profiler::CreateXprofThreadPoolExecutor("parquet_writer",
                                                             1);
}

struct ParquetRecordConsumer::Impl {
  Impl(const internal::FieldIndices& indices, absl::string_view file_path,
       tensorflow::profiler::ExecutorFactoryRef executor_factory,
       ParquetExportOptions options,
       std::shared_ptr<arrow::Schema> arrow_schema,
       std::shared_ptr<internal::TslArrowOutputStream> outfile,
       std::unique_ptr<parquet::arrow::FileWriter> file_writer)
      : file_path(file_path),
        options(std::move(options)),
        indices(indices),
        executor(executor_factory()),
        arrow_schema(std::move(arrow_schema)),
        outfile(std::move(outfile)),
        file_writer(std::move(file_writer)),
        batches{Batch(indices, this->options.batch_size),
                Batch(indices, this->options.batch_size)} {}

  ~Impl() { executor->JoinAll(); }

  absl::StatusOr<StepControl> Consume(Record& record) {
    // If enough records have been consumed, stop the consumer.
    uint64_t seq = record_count.load(std::memory_order_relaxed);
    while (seq != options.max_record_count.value_or(
                      std::numeric_limits<uint64_t>::max()) &&
           !record_count.compare_exchange_weak(seq, seq + 1,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed))
      continue;
    if (options.max_record_count.has_value() &&
        seq == *options.max_record_count) {
      return StepControl::kStop;
    }
    if (seq == std::numeric_limits<uint64_t>::max()) {
      return absl::InternalError(
          "At most uint64_t::max records can be exported to Parquet.");
    }

    // Determine the next batch and index within the batch for the record.
    //
    // `epoch` is incremented in `WriteBatch` using `memory_order_release`.
    // It is loaded here using `memory_order_acquire` to guarantee that we
    // see the updated value.
    uint64_t curr_epoch = epoch.load(std::memory_order_acquire);
    const uint64_t next_epoch = seq / options.batch_size;
    const uint32_t next_index = seq % options.batch_size;
    Batch& batch = batches[next_epoch % 2];

    // We must wait if we are switching epochs and the previous records in the
    // new epoch have not been written yet.
    if (next_epoch >= 2) {
      while (curr_epoch <= next_epoch - 2) {
        epoch.wait(curr_epoch, std::memory_order_relaxed);
        curr_epoch = epoch.load(std::memory_order_acquire);
      }
    }

    const bool failed_before = failed.load(std::memory_order_acquire);
    if (!failed_before) batch.Fill(record, next_index);
    if (batch.ready_count.fetch_add(1, std::memory_order_release) + 1 ==
        options.batch_size) {
      WaitIfEpochIsTooFarBehind(next_epoch);
      executor->Execute([this, &batch] { WriteBatch(batch); });
    }

    if (failed_before) {
      // After `failed` is set to true, `write_status` is never assigned a
      // value. `failed.store` is called with `memory_order_release`, so any
      // previous writes are guaranteed to be visible to this thread.
      return write_status;
    }
    return StepControl::kContinue;
  }

  absl::Status Finalize(const absl::StatusOr<ParseStatus>& result) {
    // No other threads are running `Consume` at this point.
    const uint64_t seq = record_count.load(std::memory_order_relaxed);
    if (result.ok() && seq % options.batch_size != 0) {
      const uint64_t epoch_num = seq / options.batch_size;
      Batch& batch = batches[epoch_num % 2];
      WaitIfEpochIsTooFarBehind(epoch_num);
      executor->Execute([this, &batch] { WriteBatch(batch); });
    }
    executor->JoinAll();
    TF_RETURN_IF_ERROR(result.status());
    // All calls to `WriteBatch` have completed. `write_status` is not being
    // written to by any other threads.
    write_status.Update(internal::ToAbslStatus(file_writer->Close()));
    write_status.Update(internal::ToAbslStatus(outfile->Close()));
    return write_status;
  }

  void WaitIfEpochIsTooFarBehind(uint64_t epoch_num) {
    // `executor` does not guarantee to process in order. Therefore, we must
    // wait for the previous batch to be written before we can write the
    // current batch.
    //
    // We must not block executor thread on this condition. Because if there
    // is only one executor thread, this will cause a deadlock.
    //
    // `epoch` is incremented using `memory_order_release` and we read it here
    // using `memory_order_acquire`. This guarantees that we see updates to
    // `write_status`. Therefore, the next executor thread will see the updated
    // `write_status` as well.
    epoch.wait(epoch_num - 1, std::memory_order_acquire);
  }

  void WriteBatch(Batch& batch) {
    // At most one thread is executing this code at a time.
    //
    // `ready_count` is increased using `memory_order_release` and we read it
    // here using `memory_order_acquire`. This guarantees that we will see any
    // previous writes to the batch columns.
    const uint32_t ready_count =
        batch.ready_count.load(std::memory_order_acquire);
    if (!failed.load(std::memory_order_relaxed)) {
      absl::StatusOr<std::vector<std::shared_ptr<arrow::Array>>> arrays =
          batch.ToArrowArrays(ready_count);
      if (arrays.ok()) {
        const std::shared_ptr<arrow::Table> table =
            arrow::Table::Make(arrow_schema, std::move(*arrays), ready_count);
        write_status = internal::ToAbslStatus(
            file_writer->WriteTable(*table, ready_count));
        if (!write_status.ok()) failed.store(true, std::memory_order_release);
      } else {
        write_status = arrays.status();
        failed.store(true, std::memory_order_release);
      }
    }
    batch.ready_count.store(0, std::memory_order_relaxed);
    epoch.fetch_add(1, std::memory_order_release);
    epoch.notify_all();
  }

  const std::string file_path;
  const ParquetExportOptions options;
  const internal::FieldIndices indices;
  const std::unique_ptr<tensorflow::profiler::Executor> executor;
  const std::shared_ptr<arrow::Schema> arrow_schema;
  const std::shared_ptr<internal::TslArrowOutputStream> outfile;
  const std::unique_ptr<parquet::arrow::FileWriter> file_writer;

  std::array<Batch, 2> batches;
  std::atomic<uint64_t> record_count{0};
  std::atomic<uint64_t> epoch{0};
  std::atomic<bool> failed{false};
  absl::Status write_status = absl::OkStatus();
};

absl::StatusOr<ParquetRecordConsumer> ParquetRecordConsumer::Build(
    Schema& schema, absl::string_view file_path,
    tensorflow::profiler::ExecutorFactoryRef executor_factory,
    ParquetExportOptions options) {
  if (options.batch_size == 0) {
    return absl::InvalidArgumentError("batch_size must be positive.");
  }
  if (options.compression_level.has_value() &&
      !options.compression_type.has_value()) {
    return absl::InvalidArgumentError(
        "compression_level requires compression_type to be set.");
  }
  internal::FieldIndices indices(schema);
  std::shared_ptr<arrow::Schema> arrow_schema =
      CreateArrowSchema(CreateColumns(indices, 0));
  TF_ASSIGN_OR_RETURN(std::shared_ptr<internal::TslArrowOutputStream> outfile,
                      internal::TslArrowOutputStream::Open(file_path));
  // `parquet::WriterProperties` is defined in `parquet/properties.h` and
  // provided transitively via `parquet/arrow/writer.h`; suppress
  // `misc-include-cleaner` since `properties.h` is not exported for direct
  // inclusion.
  parquet::WriterProperties::Builder builder;  // NOLINT(misc-include-cleaner)
  if (options.compression_type.has_value()) {
    builder.compression(*options.compression_type);
    if (options.compression_level.has_value())
      builder.compression_level(*options.compression_level);
  }
  // NOLINTNEXTLINE(misc-include-cleaner)
  std::shared_ptr<parquet::WriterProperties> writer_properties =
      builder.build();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<parquet::arrow::FileWriter> file_writer,
                      internal::ToAbslStatusOr(parquet::arrow::FileWriter::Open(
                          *arrow_schema, arrow::default_memory_pool(), outfile,
                          std::move(writer_properties))));
  return ParquetRecordConsumer(std::make_unique<Impl>(
      indices, file_path, executor_factory, std::move(options),
      std::move(arrow_schema), std::move(outfile), std::move(file_writer)));
}

ParquetRecordConsumer::ParquetRecordConsumer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ParquetRecordConsumer::ParquetRecordConsumer(ParquetRecordConsumer&&) = default;

ParquetRecordConsumer& ParquetRecordConsumer::operator=(
    ParquetRecordConsumer&&) = default;

ParquetRecordConsumer::~ParquetRecordConsumer() = default;

absl::StatusOr<StepControl> ParquetRecordConsumer::Consume(Record& record) {
  return impl_->Consume(record);
}

absl::Status ParquetRecordConsumer::Finalize(
    const absl::StatusOr<ParseStatus>& result) {
  return impl_->Finalize(result);
}

}  // namespace xprof::events_db
