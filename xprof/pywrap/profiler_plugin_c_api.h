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

#ifndef XPROF_PYWRAP_PROFILER_PLUGIN_C_API_H_
#define XPROF_PYWRAP_PROFILER_PLUGIN_C_API_H_

#include <stdbool.h>
#include <stddef.h>

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

EXPORT_C void InitializeProfiler();

// Returns memory allocated in standard c-api strings. Free using FreeString.
EXPORT_C void FreeString(char* str);

// Error return strings are allocated with strdup and must be freed with
// FreeString. If the functions return something other than void, NULL return
// means success, and on failure return the error string.

EXPORT_C char* Trace(const char* service_addr, const char* logdir,
            const char* worker_list, bool include_dataset_ops, int duration_ms,
            int num_tracing_attempts, const char** option_keys,
            const char** option_string_vals, const int* option_int_vals,
            const bool* option_bool_vals, const int* option_types,
            int num_options);

EXPORT_C char* Monitor(const char* service_addr, int duration_ms,
                       int monitoring_level, bool display_timestamp,
                       char** result_content);

EXPORT_C char* StartContinuousProfiling(
    const char* service_addr, const char** option_keys,
    const char** option_string_vals, const int* option_int_vals,
    const bool* option_bool_vals, const int* option_types, int num_options);

EXPORT_C char* StopContinuousProfiling(const char* service_addr);

EXPORT_C char* GetSnapshot(const char* service_addr, const char* logdir);

EXPORT_C char* XSpaceToToolsData(
    const char** xspace_paths, size_t num_xspace_paths, const char* tool_name,
    const char** option_keys, const char** option_string_vals,
    const int* option_int_vals, const bool* option_bool_vals,
    const int* option_types, int num_options, char** result_data,
    size_t* result_data_size, bool* success);

EXPORT_C char* XSpaceToToolsDataFromByteString(
    const char** xspace_strings, const size_t* xspace_string_sizes,
    const char** xspace_paths, size_t num_xspaces, const char* tool_name,
    const char** option_keys, const char** option_string_vals,
    const int* option_int_vals, const bool* option_bool_vals,
    const int* option_types, int num_options, char** result_data,
    size_t* result_data_size, bool* success);

EXPORT_C void StartGrpcServer(int port, int max_concurrent_requests);

EXPORT_C void InitializeStubs(const char* worker_service_addresses);

#ifdef __cplusplus
}
#endif

#endif  // XPROF_PYWRAP_PROFILER_PLUGIN_C_API_H_
