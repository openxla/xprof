/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"

namespace {

using ::testing::HasSubstr;

// The C API rejects a NULL/empty tool_name with a clean error instead of
// dereferencing it downstream. Run under --config=asan to prove it.

TEST(ProfilerPluginCApiTest, XSpaceToToolsDataNullToolNameReturnsError) {
  char* result_data = nullptr;
  size_t result_data_size = 0;
  bool success = true;

  char* err = XSpaceToToolsData(
      /*xspace_paths=*/nullptr, /*num_xspace_paths=*/0, /*tool_name=*/nullptr,
      /*option_keys=*/nullptr, /*option_string_vals=*/nullptr,
      /*option_int_vals=*/nullptr, /*option_bool_vals=*/nullptr,
      /*option_types=*/nullptr, /*num_options=*/0, &result_data,
      &result_data_size, &success);

  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("tool_name must be a non-empty"));
  EXPECT_FALSE(success);
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, XSpaceToToolsDataEmptyToolNameReturnsError) {
  char* result_data = nullptr;
  size_t result_data_size = 0;
  bool success = true;

  char* err = XSpaceToToolsData(
      /*xspace_paths=*/nullptr, /*num_xspace_paths=*/0, /*tool_name=*/"",
      /*option_keys=*/nullptr, /*option_string_vals=*/nullptr,
      /*option_int_vals=*/nullptr, /*option_bool_vals=*/nullptr,
      /*option_types=*/nullptr, /*num_options=*/0, &result_data,
      &result_data_size, &success);

  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("tool_name must be a non-empty"));
  EXPECT_FALSE(success);
  FreeString(err);
}

TEST(ProfilerPluginCApiTest,
     XSpaceToToolsDataFromByteStringNullToolNameReturnsError) {
  char* result_data = nullptr;
  size_t result_data_size = 0;
  bool success = true;

  char* err = XSpaceToToolsDataFromByteString(
      /*xspace_strings=*/nullptr, /*xspace_string_sizes=*/nullptr,
      /*xspace_paths=*/nullptr, /*num_xspaces=*/0, /*tool_name=*/nullptr,
      /*option_keys=*/nullptr, /*option_string_vals=*/nullptr,
      /*option_int_vals=*/nullptr, /*option_bool_vals=*/nullptr,
      /*option_types=*/nullptr, /*num_options=*/0, &result_data,
      &result_data_size, &success);

  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("tool_name must be a non-empty"));
  EXPECT_FALSE(success);
  FreeString(err);
}

TEST(ProfilerPluginCApiTest,
     XSpaceToToolsDataFromByteStringEmptyToolNameReturnsError) {
  char* result_data = nullptr;
  size_t result_data_size = 0;
  bool success = true;

  char* err = XSpaceToToolsDataFromByteString(
      /*xspace_strings=*/nullptr, /*xspace_string_sizes=*/nullptr,
      /*xspace_paths=*/nullptr, /*num_xspaces=*/0, /*tool_name=*/"",
      /*option_keys=*/nullptr, /*option_string_vals=*/nullptr,
      /*option_int_vals=*/nullptr, /*option_bool_vals=*/nullptr,
      /*option_types=*/nullptr, /*num_options=*/0, &result_data,
      &result_data_size, &success);

  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("tool_name must be a non-empty"));
  EXPECT_FALSE(success);
  FreeString(err);
}

// The sibling profiling-control entry points reject NULL/empty required args
// (service_addr) and NULL optional args (logdir/worker_list) with a clean error
// before any downstream use.

TEST(ProfilerPluginCApiTest, TraceNullServiceAddrReturnsError) {
  char* err = Trace(/*service_addr=*/nullptr, /*logdir=*/"", /*worker_list=*/"",
                    /*include_dataset_ops=*/false, /*duration_ms=*/1,
                    /*num_tracing_attempts=*/1, /*option_keys=*/nullptr,
                    /*option_string_vals=*/nullptr, /*option_int_vals=*/nullptr,
                    /*option_bool_vals=*/nullptr, /*option_types=*/nullptr,
                    /*num_options=*/0);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, TraceEmptyServiceAddrReturnsError) {
  char* err = Trace(/*service_addr=*/"", /*logdir=*/"", /*worker_list=*/"",
                    /*include_dataset_ops=*/false, /*duration_ms=*/1,
                    /*num_tracing_attempts=*/1, /*option_keys=*/nullptr,
                    /*option_string_vals=*/nullptr, /*option_int_vals=*/nullptr,
                    /*option_bool_vals=*/nullptr, /*option_types=*/nullptr,
                    /*num_options=*/0);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, TraceNullLogdirReturnsError) {
  char* err = Trace(/*service_addr=*/"localhost:1234", /*logdir=*/nullptr,
                    /*worker_list=*/"", /*include_dataset_ops=*/false,
                    /*duration_ms=*/1, /*num_tracing_attempts=*/1,
                    /*option_keys=*/nullptr, /*option_string_vals=*/nullptr,
                    /*option_int_vals=*/nullptr, /*option_bool_vals=*/nullptr,
                    /*option_types=*/nullptr, /*num_options=*/0);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("logdir must be non-NULL"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, TraceNullWorkerListReturnsError) {
  char* err = Trace(/*service_addr=*/"localhost:1234", /*logdir=*/"",
                    /*worker_list=*/nullptr, /*include_dataset_ops=*/false,
                    /*duration_ms=*/1, /*num_tracing_attempts=*/1,
                    /*option_keys=*/nullptr, /*option_string_vals=*/nullptr,
                    /*option_int_vals=*/nullptr, /*option_bool_vals=*/nullptr,
                    /*option_types=*/nullptr, /*num_options=*/0);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("worker_list must be non-NULL"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, MonitorNullServiceAddrReturnsError) {
  char* result_content = reinterpret_cast<char*>(0x1);
  char* err = Monitor(/*service_addr=*/nullptr, /*duration_ms=*/1,
                      /*monitoring_level=*/1, /*display_timestamp=*/false,
                      &result_content);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  EXPECT_EQ(result_content, nullptr);
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, MonitorEmptyServiceAddrReturnsError) {
  char* result_content = nullptr;
  char* err = Monitor(/*service_addr=*/"", /*duration_ms=*/1,
                      /*monitoring_level=*/1, /*display_timestamp=*/false,
                      &result_content);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest,
     StartContinuousProfilingNullServiceAddrReturnsError) {
  char* err = StartContinuousProfiling(
      /*service_addr=*/nullptr, /*option_keys=*/nullptr,
      /*option_string_vals=*/nullptr, /*option_int_vals=*/nullptr,
      /*option_bool_vals=*/nullptr, /*option_types=*/nullptr,
      /*num_options=*/0);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest,
     StopContinuousProfilingNullServiceAddrReturnsError) {
  char* err = StopContinuousProfiling(/*service_addr=*/nullptr);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, GetSnapshotNullServiceAddrReturnsError) {
  char* err = GetSnapshot(/*service_addr=*/nullptr, /*logdir=*/"");
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("service_addr must be a non-empty"));
  FreeString(err);
}

TEST(ProfilerPluginCApiTest, GetSnapshotNullLogdirReturnsError) {
  char* err =
      GetSnapshot(/*service_addr=*/"localhost:1234", /*logdir=*/nullptr);
  ASSERT_NE(err, nullptr);
  EXPECT_THAT(err, HasSubstr("logdir must be non-NULL"));
  FreeString(err);
}

// InitializeStubs returns void, so the fail-closed behavior is "log and return
// early without dereferencing". A NULL deref would crash under asan/ubsan; a
// clean return is the pass condition.
TEST(ProfilerPluginCApiTest, InitializeStubsNullDoesNotCrash) {
  InitializeStubs(/*worker_service_addresses=*/nullptr);
  SUCCEED();
}

#ifdef EMBEDDED_FEATURES_ENABLED
extern "C" void* CreateLloAnalysis(const char* filename);

// CreateLloAnalysis passes `filename` to std::ifstream(const char*); NULL is
// UB. The guard returns a null handle instead (only compiled in embedded
// builds).
TEST(ProfilerPluginCApiTest, CreateLloAnalysisNullFilenameReturnsNull) {
  EXPECT_EQ(CreateLloAnalysis(/*filename=*/nullptr), nullptr);
}
#endif  // EMBEDDED_FEATURES_ENABLED

}  // namespace
