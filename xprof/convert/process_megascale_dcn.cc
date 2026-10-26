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
#include "xprof/convert/process_megascale_dcn.h"

#include <climits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/data_table_utils.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::MicroToMilli;
using tsl::profiler::SafeDivide;

static constexpr double kbandwidthConversionFactor =
    /*bytes_to_gigabits=*/8E-9 / /*us_to_seconds=*/1E-6;

DataTable GetMegaScaleDataTable(const DcnSlackAnalysis& dcn_slack_analysis) {
  DataTable data_table;

  std::vector<std::vector<std::string>> kColumns = {
      {"rendezvous_name", "string", "Rendezvous Name"},
      {"recv_op_name", "string", "Recv Op Name"},
      {"send_op_name", "string", "Send Op Name"},
      {"transfer_type", "string", "Transfer Type"},
      {"slack_time", "number", "Slack Time (ms)"},
      {"host_stall", "string", "Host Stall (ms)"},
      {"observed_duration", "number", "Observed Duration (ms)"},
      {"send_stall", "number", "Send Stall (ms)"},
      {"send_done_stall", "number", "SendDone Stall (ms)"},
      {"recv_stall", "number", "Recv Stall (ms)"},
      {"recv_done_stall", "number", "RecvDone Stall (ms)"},
      {"stall_duration", "number", "Stall Duration (ms)"},
      {"total_stall", "number", "Aggregated Total Stall (ms)"},
      {"occurrences", "number", "Occurrences"},
      {"net_tx_bytes", "number", "Data Transmitted Size"},
      {"required_bandwidth", "number", "Required Bandwidth (Gbps)"},
  };

  for (const auto& column : kColumns) {
    data_table.AddColumn(TableColumn(column[0], column[1], column[2]));
  }

  for (auto& slack : dcn_slack_analysis.dcn_slack_summary()) {
    TableRow* row = data_table.AddRow();
    row->AddTextCell(slack.rendezvous());
    row->AddTextCell(slack.recv_op_name());
    row->AddTextCell(slack.send_op_name());
    row->AddTextCell(slack.transfer_type());
    row->AddNumberCell(MicroToMilli(slack.slack_us()));
    if (slack.host_events_count() > 0) {
      row->AddTextCell(absl::StrCat(MicroToMilli(slack.host_stall_us())));
    } else {
      row->AddTextCell("-");
    }
    row->AddNumberCell(MicroToMilli(slack.observed_duration_us()));
    row->AddNumberCell(MicroToMilli(slack.send_duration_us()));
    row->AddNumberCell(MicroToMilli(slack.send_done_duration_us()));
    row->AddNumberCell(MicroToMilli(slack.recv_duration_us()));
    row->AddNumberCell(MicroToMilli(slack.recv_done_duration_us()));
    row->AddNumberCell(MicroToMilli(slack.stall_duration_us()));
    row->AddNumberCell(
        MicroToMilli(slack.stall_duration_us() * slack.occurrences()));
    row->AddNumberCell(slack.occurrences());
    row->AddBytesCell(slack.bytes_transmitted_over_network());
    if (slack.slack_us() == 0) {
      row->AddNumberCell(INT_MAX);
    } else {
      row->AddNumberCell(
          SafeDivide(slack.bytes_transmitted_over_network(), slack.slack_us()) *
          kbandwidthConversionFactor);
    }
  }
  return data_table;
}

std::string GenerateMegaScaleJson(const DcnSlackAnalysis& dcn_slack_analysis) {
  DataTable data_table = GetMegaScaleDataTable(dcn_slack_analysis);
  return absl::StrCat("[", data_table.ToJson(), "]");
}
}  // namespace profiler
}  // namespace tensorflow
