/* Copyright 2026 The XProf Authors. All Rights Reserved.

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

#include "xprof/pywrap/profiler_plugin_c_api.h"

#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/worker/stub_factory.h"
#include "xprof/pywrap/profiler_plugin_impl.h"

namespace {

using ::tensorflow::profiler::ToolOptions;

ToolOptions ToolOptionsFromCArrays(const char** option_keys,
                                   const char** option_string_vals,
                                   const int* option_int_vals,
                                   const bool* option_bool_vals,
                                   const int* option_types, int num_options) {
  ToolOptions map;
  if (num_options == 0 || !option_keys || !option_types) return map;

  for (int i = 0; i < num_options; ++i) {
    if (!option_keys[i]) continue;
    std::string key = option_keys[i];
    int type = option_types[i];

    if (type == 0 && option_bool_vals) {
      map.emplace(key, option_bool_vals[i]);
    } else if (type == 1 && option_int_vals) {
      map.emplace(key, option_int_vals[i]);
    } else if (type == 2 && option_string_vals && option_string_vals[i]) {
      map.emplace(key, std::string(option_string_vals[i]));
    }
  }
  return map;
}

char* StatusToCError(const absl::Status& status) {
  if (status.ok()) return nullptr;
  return strdup(status.message().data());
}

}  // namespace

extern "C" {
EXPORT_C void InitializeProfiler() {
  int argc = 1;
  static char arg0[] = "profiler_plugin";
  static char* argv[] = {arg0};
  char** argv_ptr = argv;
}

EXPORT_C void FreeString(char* str) { free(str); }

EXPORT_C char* Trace(const char* service_addr, const char* logdir,
                     const char* worker_list, bool include_dataset_ops,
                     int duration_ms, int num_tracing_attempts,
                     const char** option_keys, const char** option_string_vals,
                     const int* option_int_vals, const bool* option_bool_vals,
                     const int* option_types, int num_options) {
  ToolOptions tool_options =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);
  absl::Status status = tsl::profiler::CaptureRemoteTrace(
      service_addr, logdir, worker_list, include_dataset_ops, duration_ms,
      num_tracing_attempts, tool_options);
  return StatusToCError(status);
}

EXPORT_C char* Monitor(const char* service_addr, int duration_ms,
                       int monitoring_level, bool display_timestamp,
                       char** result_content) {
  if (result_content) *result_content = nullptr;
  std::string content;
  absl::Status status = xprof::pywrap::Monitor(
      service_addr, duration_ms, monitoring_level, display_timestamp, &content);
  if (!status.ok()) {
    return StatusToCError(status);
  }
  if (result_content) {
    *result_content = strdup(content.c_str());
  }
  return nullptr;
}

EXPORT_C char* StartContinuousProfiling(
    const char* service_addr, const char** option_keys,
    const char** option_string_vals, const int* option_int_vals,
    const bool* option_bool_vals, const int* option_types, int num_options) {
  ToolOptions tool_options =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);
  absl::Status status =
      xprof::pywrap::StartContinuousProfiling(service_addr, tool_options);
  return StatusToCError(status);
}

EXPORT_C char* StopContinuousProfiling(const char* service_addr) {
  absl::Status status = xprof::pywrap::StopContinuousProfiling(service_addr);
  return StatusToCError(status);
}

EXPORT_C char* GetSnapshot(const char* service_addr, const char* logdir) {
  absl::Status status = xprof::pywrap::GetSnapshot(service_addr, logdir);
  return StatusToCError(status);
}

EXPORT_C char* XSpaceToToolsData(
    const char** xspace_paths, size_t num_xspace_paths, const char* tool_name,
    const char** option_keys, const char** option_string_vals,
    const int* option_int_vals, const bool* option_bool_vals,
    const int* option_types, int num_options, char** result_data,
    size_t* result_data_size, bool* success) {
  if (result_data) *result_data = nullptr;
  if (result_data_size) *result_data_size = 0;
  if (success) *success = false;

  std::vector<std::string> paths;
  paths.reserve(num_xspace_paths);
  for (size_t i = 0; i < num_xspace_paths; ++i) {
    if (xspace_paths[i]) {
      paths.push_back(xspace_paths[i]);
    }
  }

  ToolOptions tool_options =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);

  absl::StatusOr<std::pair<std::string, bool>> result =
      xprof::pywrap::XSpaceToToolsData(std::move(paths), tool_name,
                                       tool_options);

  if (!result.ok()) {
    return StatusToCError(result.status());
  }

  if (result_data) {
    // We allocate with malloc since we expect caller to free, but strdup
    // does not support binary data (it stops at null termination).
    *result_data = (char*)malloc(result->first.size());
    memcpy(*result_data, result->first.data(), result->first.size());
  }
  if (result_data_size) {
    *result_data_size = result->first.size();
  }
  if (success) {
    *success = result->second;
  }
  return nullptr;
}

EXPORT_C char* XSpaceToToolsDataFromByteString(
    const char** xspace_strings, const size_t* xspace_string_sizes,
    const char** xspace_paths, size_t num_xspaces, const char* tool_name,
    const char** option_keys, const char** option_string_vals,
    const int* option_int_vals, const bool* option_bool_vals,
    const int* option_types, int num_options, char** result_data,
    size_t* result_data_size, bool* success) {
  if (result_data) *result_data = nullptr;
  if (result_data_size) *result_data_size = 0;
  if (success) *success = false;

  std::vector<std::string> strings;
  std::vector<std::string> paths;
  strings.reserve(num_xspaces);
  paths.reserve(num_xspaces);

  for (size_t i = 0; i < num_xspaces; ++i) {
    if (xspace_strings[i]) {
      strings.push_back(std::string(xspace_strings[i], xspace_string_sizes[i]));
    } else {
      strings.push_back("");
    }
    if (xspace_paths[i]) {
      paths.push_back(xspace_paths[i]);
    } else {
      paths.push_back("");
    }
  }

  ToolOptions tool_options =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);

  absl::StatusOr<std::pair<std::string, bool>> result =
      xprof::pywrap::XSpaceToToolsDataFromByteString(
          std::move(strings), std::move(paths), tool_name, tool_options);

  if (!result.ok()) {
    return StatusToCError(result.status());
  }

  if (result_data) {
    *result_data = (char*)malloc(result->first.size());
    memcpy(*result_data, result->first.data(), result->first.size());
  }
  if (result_data_size) {
    *result_data_size = result->first.size();
  }
  if (success) {
    *success = result->second;
  }
  return nullptr;
}

EXPORT_C void StartGrpcServer(int port, int max_concurrent_requests) {
  xprof::pywrap::StartGrpcServer(port, max_concurrent_requests);
}

EXPORT_C void InitializeStubs(const char* worker_service_addresses) {
  xprof::profiler::InitializeStubs(worker_service_addresses);
}
// Provide a minimal PyInit for Python import resolution
// This allows loaders to properly satisfy `DT_NEEDED` dependencies before we
// use standard ctypes bindings later.
#include <Python.h>

static struct PyModuleDef profiler_plugin_c_api_module = {
    PyModuleDef_HEAD_INIT, "profiler_plugin_c_api", NULL, -1, NULL};

PyMODINIT_FUNC PyInit_profiler_plugin_c_api(void) {
  return PyModule_Create(&profiler_plugin_c_api_module);
}

}  // extern "C"
