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

#include "xprof/convert/grpc_metrics/grpc_metrics_utils.h"

#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"

namespace xprof::grpc_metrics {
namespace {

// Adds a straggler with the given flow id and server latency to `db`.
xprof::GrpcStraggler* AddStraggler(absl::string_view flow_id,
                                   double server_latency_ms,
                                   xprof::GrpcMetricsDatabase* db) {
  xprof::GrpcStraggler* straggler = db->add_top_stragglers();
  straggler->set_chunk_id(flow_id);
  straggler->set_grpc_server_latency_ms(server_latency_ms);
  return straggler;
}

// Why: the per-host step must keep the totals of the parser and give each
// straggler the server latency recorded on the same host.
// What: checks the totals, the local match and the unmatched latencies.
TEST(FinalizeHostGrpcMetricsTest, StoresTotalsAndMatchesLocalLatencies) {
  // Setup.
  xprof::GrpcMetricsDatabase host_db;
  AddStraggler("flow_a", 0.0, &host_db);
  AddStraggler("flow_b", 0.0, &host_db);
  GrpcMetricsParseResults results;
  results.total_network_stall_ms = 12.5;
  results.total_compute_overlap_ms = 3.0;
  results.server_latencies["flow_a"] = 4.0;
  results.server_latencies["flow_c"] = 7.0;

  // Execution.
  FinalizeHostGrpcMetrics(std::move(results), &host_db);

  // Verification.
  EXPECT_DOUBLE_EQ(host_db.total_network_stall_ms(), 12.5);
  EXPECT_DOUBLE_EQ(host_db.total_compute_overlap_ms(), 3.0);
  ASSERT_EQ(host_db.top_stragglers_size(), 2);
  EXPECT_DOUBLE_EQ(host_db.top_stragglers(0).grpc_server_latency_ms(), 4.0);
  EXPECT_DOUBLE_EQ(host_db.top_stragglers(1).grpc_server_latency_ms(), 0.0);
  ASSERT_EQ(host_db.unmatched_server_latencies_size(), 1);
  EXPECT_DOUBLE_EQ(host_db.unmatched_server_latencies().at("flow_c"), 7.0);
}

// Why: one host must not keep more stragglers than the local limit.
// What: checks that the first kMaxLocalStragglers stragglers are kept.
TEST(FinalizeHostGrpcMetricsTest, KeepsAtMostMaxLocalStragglers) {
  // Setup.
  xprof::GrpcMetricsDatabase host_db;
  const int straggler_count = kMaxLocalStragglers + 2;
  for (int i = 0; i < straggler_count; ++i) {
    AddStraggler(absl::StrCat("flow_", i), 0.0, &host_db);
  }

  // Execution.
  FinalizeHostGrpcMetrics(GrpcMetricsParseResults(), &host_db);

  // Verification.
  ASSERT_EQ(host_db.top_stragglers_size(), kMaxLocalStragglers);
  EXPECT_EQ(host_db.top_stragglers(kMaxLocalStragglers - 1).chunk_id(),
            absl::StrCat("flow_", kMaxLocalStragglers - 1));
}

// Why: hosts are merged by summing their totals and appending their lists.
// What: checks the sums, the straggler order, the time series and the
// unmatched latencies, where the host that is merged last wins.
TEST(MergeHostGrpcMetricsTest, SumsTotalsAndAppendsLists) {
  // Setup.
  xprof::GrpcMetricsDatabase merged_db;
  merged_db.set_total_network_stall_ms(1.0);
  merged_db.set_total_compute_overlap_ms(2.0);
  AddStraggler("flow_a", 0.0, &merged_db);
  (*merged_db.mutable_unmatched_server_latencies())["flow_x"] = 1.0;
  xprof::GrpcMetricsDatabase host_db;
  host_db.set_total_network_stall_ms(0.5);
  host_db.set_total_compute_overlap_ms(0.25);
  AddStraggler("flow_b", 0.0, &host_db);
  host_db.add_chunk_latency_series()->set_value(9.0);
  (*host_db.mutable_unmatched_server_latencies())["flow_x"] = 2.0;
  (*host_db.mutable_unmatched_server_latencies())["flow_y"] = 3.0;

  // Execution.
  MergeHostGrpcMetrics(std::move(host_db), &merged_db);

  // Verification.
  EXPECT_DOUBLE_EQ(merged_db.total_network_stall_ms(), 1.5);
  EXPECT_DOUBLE_EQ(merged_db.total_compute_overlap_ms(), 2.25);
  ASSERT_EQ(merged_db.top_stragglers_size(), 2);
  EXPECT_EQ(merged_db.top_stragglers(0).chunk_id(), "flow_a");
  EXPECT_EQ(merged_db.top_stragglers(1).chunk_id(), "flow_b");
  ASSERT_EQ(merged_db.chunk_latency_series_size(), 1);
  EXPECT_DOUBLE_EQ(merged_db.chunk_latency_series(0).value(), 9.0);
  ASSERT_EQ(merged_db.unmatched_server_latencies_size(), 2);
  EXPECT_DOUBLE_EQ(merged_db.unmatched_server_latencies().at("flow_x"), 2.0);
  EXPECT_DOUBLE_EQ(merged_db.unmatched_server_latencies().at("flow_y"), 3.0);
}

// Why: a server latency recorded on another host must only fill stragglers
// that have no server latency yet.
// What: checks the cross-host match and that the unmatched map is cleared.
TEST(FinalizeMergedGrpcMetricsTest, FillsOnlyMissingServerLatencies) {
  // Setup.
  xprof::GrpcMetricsDatabase merged_db;
  AddStraggler("flow_a", 0.0, &merged_db);
  AddStraggler("flow_b", 2.0, &merged_db);
  AddStraggler("flow_c", 0.0, &merged_db);
  (*merged_db.mutable_unmatched_server_latencies())["flow_a"] = 5.0;
  (*merged_db.mutable_unmatched_server_latencies())["flow_b"] = 9.0;

  // Execution.
  FinalizeMergedGrpcMetrics(&merged_db);

  // Verification.
  ASSERT_EQ(merged_db.top_stragglers_size(), 3);
  EXPECT_DOUBLE_EQ(merged_db.top_stragglers(0).grpc_server_latency_ms(), 5.0);
  EXPECT_DOUBLE_EQ(merged_db.top_stragglers(1).grpc_server_latency_ms(), 2.0);
  EXPECT_DOUBLE_EQ(merged_db.top_stragglers(2).grpc_server_latency_ms(), 0.0);
  EXPECT_EQ(merged_db.unmatched_server_latencies_size(), 0);
}

// Why: the page reads proto field names and relies on unset fields being left
// out, while explicitly set optional fields are printed even when zero.
// What: checks the JSON of a small database.
TEST(ConvertGrpcMetricsToJsonTest, UsesProtoNamesAndSkipsDefaults) {
  // Setup.
  xprof::GrpcMetricsDatabase db;
  db.set_total_network_stall_ms(1.5);
  xprof::GrpcStraggler* straggler = AddStraggler("flow_a", 0.0, &db);
  straggler->set_slack_time_ms(0.0);

  // Execution.
  std::string json;
  const absl::Status status = ConvertGrpcMetricsToJson(db, &json);

  // Verification.
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(json,
            "{\"top_stragglers\":[{\"chunk_id\":\"flow_a\",\"slack_time_ms\":0}"
            "],\"total_network_stall_ms\":1.5}");
}

}  // namespace
}  // namespace xprof::grpc_metrics
