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

#ifndef XPROF_CONVERT_EVENTS_DB_PYTHON_PYWRAP_EVENTS_DB_C_API_H_
#define XPROF_CONVERT_EVENTS_DB_PYTHON_PYWRAP_EVENTS_DB_C_API_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EXPORT_C
#ifdef _WIN32
#define EXPORT_C __declspec(dllexport)
#else
#define EXPORT_C __attribute__((visibility("default")))
#endif
#endif

// Frees memory allocated for strings returned by the C API (e.g. error
// strings).
EXPORT_C void XProfEventsDbFreeString(char* str);

// Converts an XSpace trace file to an Events DB Parquet file.
//
// Parameters:
//   input_path: Path to the input XSpace/trace file (e.g. `.xplane.pb`).
//   output_path: Destination path for the generated `.parquet` file.
//   batch_size: Number of records per Parquet batch (`0` for unspecified).
//   compression_type: Codec name (`"SNAPPY"` or `"ZSTD"`), or `NULL` for
//     unspecified.
//   compression_level: Compression level (`-1` for unspecified).
//   max_record_count: Max records to write (`0` for unlimited).
//   out_error_message: Optional pointer to a `char*`. If not `NULL`, receives a
//     heap-allocated error string on failure that must be freed by the caller
//     using `XProfEventsDbFreeString`. On success, `*out_error_message` is set
//     to `NULL`. If memory allocation fails on error, `*out_error_message` is
//     set to `NULL`.
//
// Returns:
//   `true` on success, or `false` on failure.
EXPORT_C bool XProfEventsDbXSpaceToParquet(
    const char* input_path, const char* output_path, uint32_t batch_size,
    const char* compression_type, int compression_level,
    uint64_t max_record_count, char** out_error_message);

#ifdef __cplusplus
}
#endif

#endif  // XPROF_CONVERT_EVENTS_DB_PYTHON_PYWRAP_EVENTS_DB_C_API_H_
