/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "xprof/convert/tpu_input_pipeline_analysis_constants.h"

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

constexpr absl::string_view kProfileAllHostsDoc =
    "https://cloud.google.com/tpu/docs/troubleshooting/troubleshoot-multislice";
constexpr absl::string_view kSparseCoreV0Name = "SparseCoreV0";
constexpr absl::string_view kSparseCoreV0ComputeTimeMsId = "scv0ComputeTimeMs";
constexpr absl::string_view kSparseCoreV0ComputeTimeMsLabel =
    "SparseCoreV0 compute (in ms)";
constexpr absl::string_view kSparseCoreV0InfeedTimeMsId = "scv0InfeedTimeMs";
constexpr absl::string_view kSparseCoreV0InfeedTimeMsLabel =
    "SparseCoreV0 input (in ms)";
constexpr absl::string_view kSparseCoreV0ComputeMsAverage =
    "scv0_compute_ms_average";
constexpr absl::string_view kSparseCoreV0InfeedMsAverage =
    "scv0_infeed_ms_average";

}  // namespace profiler
}  // namespace tensorflow
