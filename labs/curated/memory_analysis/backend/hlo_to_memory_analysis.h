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

#ifndef THIRD_PARTY_XPROF_LABS_CURATED_MEMORY_ANALYSIS_BACKEND_HLO_TO_MEMORY_ANALYSIS_H_
#define THIRD_PARTY_XPROF_LABS_CURATED_MEMORY_ANALYSIS_BACKEND_HLO_TO_MEMORY_ANALYSIS_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"

namespace tensorflow {
namespace profiler {

// Analyzes an HLO Proto and generates a categorized memory analysis JSON string
// containing peak memory summary, memory space breakdown, and buffer details.
// Returns an error if the HLO module configuration cannot be created or if
// preprocessing fails.
absl::StatusOr<std::string> ConvertHloProtoToMemoryAnalysisJson(
    const xla::HloProto& hlo_proto, absl::string_view module_name);

// Performs graph tracing across HLO instructions to identify all-gather weights
// and their derived buffers. Returns a set of instruction names identified as
// gathered weights. Exposed for testing.
absl::StatusOr<absl::flat_hash_set<std::string>> TraceGatheredWeights(
    const xla::HloProto& hlo_proto);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_LABS_CURATED_MEMORY_ANALYSIS_BACKEND_HLO_TO_MEMORY_ANALYSIS_H_
