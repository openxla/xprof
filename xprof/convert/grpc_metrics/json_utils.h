/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_GRPC_METRICS_JSON_UTILS_H_
#define XPROF_CONVERT_GRPC_METRICS_JSON_UTILS_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace xprof::grpc_metrics {

absl::string_view ExtractJsonValue(absl::string_view json,
                                   absl::string_view key);

std::vector<std::string> ParseJsonArray(absl::string_view array_str);

double GetNumFromJson(absl::string_view json, absl::string_view key,
                      double default_val = 0.0);

std::string GetIdFromJson(absl::string_view json, absl::string_view key);

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_JSON_UTILS_H_
