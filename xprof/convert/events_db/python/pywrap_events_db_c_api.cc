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

#include "xprof/convert/events_db/python/pywrap_events_db_c_api.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "arrow/result.h"  // from @arrow
#include "arrow/util/compression.h"  // from @arrow
#include "arrow/util/type_fwd.h"  // from @arrow
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/events_db/parquet_record_consumer.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/xspace_parser.h"

namespace {

// Returns a heap-allocated copy of the status message, or nullptr if ok.
// We avoid `strdup` here because it is a POSIX-specific function rather than
// standard C++, which causes MSVC deprecation warnings (C4996) on Windows and
// false-positive unused-include diagnostics with clangd's include-cleaner.
char* StatusToCError(const absl::Status& status) {
  if (status.ok()) return nullptr;
  const std::string message = status.ToString();
  char* const result = static_cast<char*>(malloc(message.size() + 1));
  if (result) memcpy(result, message.data(), message.size() + 1);
  return result;
}

absl::StatusOr<std::optional<arrow::Compression::type>> GetCompressionType(
    std::string_view compression_type_str) {
  if (compression_type_str.empty()) return std::nullopt;
  arrow::Result<arrow::Compression::type> codec =
      arrow::util::Codec::GetCompressionType(
          absl::AsciiStrToLower(compression_type_str));
  if (codec.ok()) return *codec;
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid compression type: '", compression_type_str, "'"));
}

absl::StatusOr<xprof::events_db::ParquetExportOptions>
BuildParquetExportOptions(uint32_t batch_size, const char* compression_type,
                          int compression_level, uint64_t max_record_count) {
  xprof::events_db::ParquetExportOptions options;
  if (batch_size > 0) options.batch_size = batch_size;
  if (max_record_count > 0) options.max_record_count = max_record_count;
  if (compression_level >= 0) options.compression_level = compression_level;
  if (compression_type && compression_type[0] != '\0') {
    TF_ASSIGN_OR_RETURN(options.compression_type,
                        GetCompressionType(compression_type));
  }
  return options;
}

absl::Status XProfEventsDbXSpaceToParquetImpl(const char* input_path,
                                              const char* output_path,
                                              uint32_t batch_size,
                                              const char* compression_type,
                                              int compression_level,
                                              uint64_t max_record_count) {
  if (!input_path || input_path[0] == '\0') {
    return absl::InvalidArgumentError("input_path must not be null or empty.");
  }
  if (!output_path || output_path[0] == '\0') {
    return absl::InvalidArgumentError("output_path must not be null or empty.");
  }

  TF_ASSIGN_OR_RETURN(
      xprof::events_db::ParquetExportOptions options,
      BuildParquetExportOptions(batch_size, compression_type, compression_level,
                                max_record_count));

  xprof::events_db::Schema schema;
  TF_ASSIGN_OR_RETURN(xprof::events_db::ParquetRecordConsumer consumer,
                      xprof::events_db::ParquetRecordConsumer::Build(
                          schema, output_path, std::move(options)));

  TF_RETURN_IF_ERROR(
      xprof::events_db::ParseXSpace(input_path, schema, consumer).status());

  return absl::OkStatus();
}

}  // namespace

extern "C" {

EXPORT_C void XProfEventsDbFreeString(char* str) { free(str); }

EXPORT_C bool XProfEventsDbXSpaceToParquet(
    const char* input_path, const char* output_path, uint32_t batch_size,
    const char* compression_type, int compression_level,
    uint64_t max_record_count, char** out_error_message) {
  const absl::Status status = XProfEventsDbXSpaceToParquetImpl(
      input_path, output_path, batch_size, compression_type, compression_level,
      max_record_count);
  if (out_error_message) *out_error_message = StatusToCError(status);
  return status.ok();
}

}  // extern "C"
