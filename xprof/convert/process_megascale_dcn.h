/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XPROF_CONVERT_PROCESS_MEGASCALE_DCN_H_
#define XPROF_CONVERT_PROCESS_MEGASCALE_DCN_H_

#include <string>

#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"

namespace tensorflow {
namespace profiler {

// Process Dcn Megascale TraceMe info.
void ProcessMegascaleDcn(XSpace* space);

DataTable GetMegaScaleDataTable(const DcnSlackAnalysis& dcn_slack_analysis);

std::string GenerateMegaScaleJson(const DcnSlackAnalysis& dcn_slack_analysis);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_PROCESS_MEGASCALE_DCN_H_
