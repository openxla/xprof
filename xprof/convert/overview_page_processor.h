/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_XPROF_CONVERT_OVERVIEW_PAGE_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_OVERVIEW_PAGE_PROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/op_stats_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {

class OverviewPageProcessor : public OpStatsProcessor {
 public:
  explicit OverviewPageProcessor(
      const tensorflow::profiler::ToolOptions& options)
      : options_(options) {}

  absl::Status ProcessCombinedOpStats(
      const tensorflow::profiler::SessionSnapshot& session_snapshot,
      const tensorflow::profiler::OpStats& combined_op_stats) override;

  bool ShouldUseWorkerService(const tensorflow::profiler::SessionSnapshot&
                                  session_snapshot) const override {
    return true;
  }

 private:
  tensorflow::profiler::ToolOptions options_;
};

REGISTER_PROFILE_PROCESSOR("overview_page", OverviewPageProcessor);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_OVERVIEW_PAGE_PROCESSOR_H_
