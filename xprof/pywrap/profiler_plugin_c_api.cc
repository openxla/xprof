#include "xprof/pywrap/profiler_plugin_c_api.h"

#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/rpc/client/capture_profile.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/worker/stub_factory.h"
#include "xprof/pywrap/profiler_plugin_impl.h"

namespace {

using ::tensorflow::profiler::ToolOptions;

absl::StatusOr<ToolOptions> ToolOptionsFromCArrays(
    const char** option_keys, const char** option_string_vals,
    const int* option_int_vals, const bool* option_bool_vals,
    const int* option_types, int num_options) {
  ToolOptions map;
  if (num_options == 0) return map;
  if (!option_keys || !option_types) {
    return absl::InvalidArgumentError(
        "option_keys and option_types must be non-NULL if num_options > 0");
  }

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
  return strdup(status.ToString().c_str());
}

}  // namespace

extern "C" {
EXPORT_C void FreeString(char* str) { free(str); }

EXPORT_C char* Trace(const char* service_addr, const char* logdir,
                     const char* worker_list, bool include_dataset_ops,
                     int duration_ms, int num_tracing_attempts,
                     const char** option_keys, const char** option_string_vals,
                     const int* option_int_vals, const bool* option_bool_vals,
                     const int* option_types, int num_options) {
  // service_addr is required non-empty; logdir and worker_list are optional.
  // Python marshals None and "" as NULL, so validate here before forwarding
  // downstream.
  if (!service_addr || service_addr[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("service_addr must be a non-empty string"));
  }
  if (!logdir) {
    return StatusToCError(
        absl::InvalidArgumentError("logdir must be non-NULL"));
  }
  if (!worker_list) {
    return StatusToCError(
        absl::InvalidArgumentError("worker_list must be non-NULL"));
  }

  auto tool_options_or =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);
  if (!tool_options_or.ok()) {
    return StatusToCError(tool_options_or.status());
  }
  ToolOptions tool_options = std::move(*tool_options_or);
  absl::Status status = tsl::profiler::CaptureRemoteTrace(
      service_addr, logdir, worker_list, include_dataset_ops, duration_ms,
      num_tracing_attempts, tool_options);
  return StatusToCError(status);
}

EXPORT_C char* Monitor(const char* service_addr, int duration_ms,
                       int monitoring_level, bool display_timestamp,
                       char** result_content) {
  if (result_content) *result_content = nullptr;

  // service_addr is a required host:port. Python marshals None and "" as NULL,
  // so reject NULL/empty before forwarding downstream.
  if (!service_addr || service_addr[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("service_addr must be a non-empty string"));
  }
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
  // service_addr is a required host:port. Python marshals None and "" as NULL,
  // so reject NULL/empty before forwarding downstream.
  if (!service_addr || service_addr[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("service_addr must be a non-empty string"));
  }

  auto tool_options_or =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);
  if (!tool_options_or.ok()) {
    return StatusToCError(tool_options_or.status());
  }
  ToolOptions tool_options = std::move(*tool_options_or);
  absl::Status status =
      xprof::pywrap::StartContinuousProfiling(service_addr, tool_options);
  return StatusToCError(status);
}

EXPORT_C char* StopContinuousProfiling(const char* service_addr) {
  // service_addr is a required host:port. Python marshals None and "" as NULL,
  // so reject NULL/empty before forwarding downstream.
  if (!service_addr || service_addr[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("service_addr must be a non-empty string"));
  }
  absl::Status status = xprof::pywrap::StopContinuousProfiling(service_addr);
  return StatusToCError(status);
}

EXPORT_C char* GetSnapshot(const char* service_addr, const char* logdir) {
  // service_addr is required non-empty; logdir is optional. Python marshals
  // None and "" as NULL, so validate here before forwarding downstream.
  if (!service_addr || service_addr[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("service_addr must be a non-empty string"));
  }
  if (!logdir) {
    return StatusToCError(
        absl::InvalidArgumentError("logdir must be non-NULL"));
  }
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

  // tool_name is required. Python marshals None and "" as NULL, so reject
  // NULL/empty before forwarding to the impl.
  if (!tool_name || tool_name[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("tool_name must be a non-empty string"));
  }

  std::vector<std::string> paths;
  paths.reserve(num_xspace_paths);
  for (size_t i = 0; i < num_xspace_paths; ++i) {
    if (xspace_paths[i]) {
      paths.push_back(xspace_paths[i]);
    }
  }

  auto tool_options_or =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);
  if (!tool_options_or.ok()) {
    return StatusToCError(tool_options_or.status());
  }
  ToolOptions tool_options = std::move(*tool_options_or);

  absl::StatusOr<std::pair<std::string, bool>> result =
      xprof::pywrap::XSpaceToToolsData(std::move(paths), tool_name,
                                       tool_options);

  if (!result.ok()) {
    return StatusToCError(result.status());
  }

  if (result_data) {
    if (!result->first.empty()) {
      // We allocate with malloc since we expect caller to free, but strdup
      // does not support binary data (it stops at null termination).
      *result_data = (char*)malloc(result->first.size());
      if (!*result_data) {
        return StatusToCError(absl::ResourceExhaustedError(
            "Failed to allocate memory for result_data"));
      }
      memcpy(*result_data, result->first.data(), result->first.size());
    } else {
      *result_data = nullptr;
    }
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

  // tool_name is required. Python marshals None and "" as NULL, so reject
  // NULL/empty before forwarding to the impl.
  if (!tool_name || tool_name[0] == '\0') {
    return StatusToCError(
        absl::InvalidArgumentError("tool_name must be a non-empty string"));
  }

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

  auto tool_options_or =
      ToolOptionsFromCArrays(option_keys, option_string_vals, option_int_vals,
                             option_bool_vals, option_types, num_options);
  if (!tool_options_or.ok()) {
    return StatusToCError(tool_options_or.status());
  }
  ToolOptions tool_options = std::move(*tool_options_or);

  absl::StatusOr<std::pair<std::string, bool>> result =
      xprof::pywrap::XSpaceToToolsDataFromByteString(
          std::move(strings), std::move(paths), tool_name, tool_options);

  if (!result.ok()) {
    return StatusToCError(result.status());
  }

  if (result_data) {
    if (!result->first.empty()) {
      *result_data = (char*)malloc(result->first.size());
      if (!*result_data) {
        return StatusToCError(absl::ResourceExhaustedError(
            "Failed to allocate memory for result_data"));
      }
      memcpy(*result_data, result->first.data(), result->first.size());
    } else {
      *result_data = nullptr;
    }
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
  // worker_service_addresses is required; this entry point returns void, so log
  // and skip initialization on NULL. An empty string is a legitimate no-op, so
  // it is not rejected.
  if (!worker_service_addresses) {
    LOG(ERROR) << "InitializeStubs: worker_service_addresses must be non-NULL; "
                  "skipping stub initialization.";
    return;
  }
  xprof::profiler::InitializeStubs(worker_service_addresses);
}
// Provide a minimal PyInit for Python import resolution
// This allows loaders to properly satisfy `DT_NEEDED` dependencies before we
// use standard ctypes bindings later.
//
// This block is only required when this translation unit is loaded as a CPython
// extension (the cc_binary .so / pytype_extension). It is compiled out for the
// standalone cc_test (which defines XPROF_PYWRAP_DISABLE_PY_MODULE_INIT) so the
// test does not need to link the CPython runtime. The shipped .so and
// pytype_extension do not define this macro and are unaffected.
#ifndef XPROF_PYWRAP_DISABLE_PY_MODULE_INIT
#include <Python.h>

static struct PyModuleDef profiler_plugin_c_api_module = {
    PyModuleDef_HEAD_INIT, "profiler_plugin_c_api", NULL, -1, NULL};

PyMODINIT_FUNC PyInit_profiler_plugin_c_api(void) {
  PyObject* m = PyModule_Create(&profiler_plugin_c_api_module);
  if (!m) return nullptr;
#ifdef Py_GIL_DISABLED
  if (PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED) != 0) {
    Py_DECREF(m);
    return nullptr;
  }
#endif
  return m;
}
#endif  // XPROF_PYWRAP_DISABLE_PY_MODULE_INIT

#ifdef EMBEDDED_FEATURES_ENABLED
EXPORT_C bool BuiltWithEmbedded() { return true; }
#include "xprof/embedded/llo_analysis/llo_analysis_c_api.h"
EXPORT_C LloAnalysisHandle CreateLloAnalysis(const char* filename) {
  // filename is required; return a null handle on NULL, which Python maps to
  // {"success": False}.
  if (!filename) {
    return nullptr;
  }
  return CreateLloAnalysisImpl(filename);
}
EXPORT_C int GetTotalInstructions(LloAnalysisHandle handle) {
  return GetTotalInstructionsImpl(handle);
}
EXPORT_C int GetUniqueRegisters(LloAnalysisHandle handle) {
  return GetUniqueRegistersImpl(handle);
}
EXPORT_C int GetNumUniqueOpcodes(LloAnalysisHandle handle) {
  return GetNumUniqueOpcodesImpl(handle);
}
EXPORT_C int GetOpcodeAtIndex(LloAnalysisHandle handle, int index) {
  return GetOpcodeAtIndexImpl(handle, index);
}
EXPORT_C int GetOpcodeCountAtIndex(LloAnalysisHandle handle, int index) {
  return GetOpcodeCountAtIndexImpl(handle, index);
}
EXPORT_C const char* GetLloDebugString(LloAnalysisHandle handle) {
  return GetLloDebugStringImpl(handle);
}
EXPORT_C void FreeLloAnalysis(LloAnalysisHandle handle) {
  FreeLloAnalysisImpl(handle);
}
#else
EXPORT_C bool BuiltWithEmbedded() { return false; }
#endif

}  // extern "C"
