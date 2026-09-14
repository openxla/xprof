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

// Example CLI to count events with zero self_time_ns in an XSpace trace.
//
// Usage:
//   ```shell
//   bazel run \
//       //xprof/convert/events_db/examples/cpp:count_zero_self_time_events \
//       -- \
//       --input_path=/path/to/trace.xplane.pb
//   ```

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tsl/platform/init_main.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/xspace_parser.h"

ABSL_FLAG(std::string, input_path, "",
          "Path to input XSpace/trace file (e.g., .xplane.pb).");

int main(int argc, char* argv[]) {
  tsl::port::InitMain(
      "Counts events with zero self_time_ns in an XSpace trace.\n"
      "Usage:\n"
      "  count_zero_self_time_events --input_path=<path> [flags]",
      &argc, &argv);

  const std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  if (positional_args.size() > 1) {
    std::cerr << "Too many command-line arguments.\n";
    return 1;
  }

  const std::string input_path = absl::GetFlag(FLAGS_input_path);
  if (input_path.empty()) {
    std::cerr << "input_path cannot be empty.\n";
    return 1;
  }

  std::cout << "Counting events with zero self_time_ns in XSpace trace...\n"
            << "  Input XSpace: '" << input_path << "'\n";

  xprof::events_db::Schema schema;
  const xprof::events_db::TypedFieldIndex<uint64_t> field =
      schema.RegisterFieldName<uint64_t>("self_time_ns");
  std::atomic<uint64_t> total_records = 0;
  std::atomic<uint64_t> zero_self_time_count = 0;

  const auto consume = [&](xprof::events_db::Record& record) {
    total_records.fetch_add(1, std::memory_order_relaxed);
    if (record.HasField(field) && record[field] == 0) {
      zero_self_time_count.fetch_add(1, std::memory_order_relaxed);
    }
    return xprof::events_db::StepControl::kContinue;
  };

  const absl::Time start_time = absl::Now();
  const absl::StatusOr<xprof::events_db::ParseStatus> parse_status =
      xprof::events_db::ParseXSpace(input_path, schema, consume);
  if (!parse_status.ok()) {
    std::cerr << "Parsing failed: " << parse_status.status() << "\n";
    return 1;
  }

  const absl::Duration elapsed = absl::Now() - start_time;
  std::cout << absl::StrFormat(
      "Successfully finished parsing in %.2fs with parse status: %s\n"
      "  Total records processed: %llu\n"
      "  Zero self_time_ns events: %llu\n",
      absl::ToDoubleSeconds(elapsed), ParseStatusToString(*parse_status),
      total_records.load(), zero_self_time_count.load());

  return 0;
}
