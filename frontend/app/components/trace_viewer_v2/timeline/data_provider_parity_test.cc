#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/json/src/json.hpp"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_stream_ingestor.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace traceviewer {
namespace {

using ::tensorflow::profiler::DeltaSeriesProtoConversionOptions;
using ::tensorflow::profiler::Device;
using ::tensorflow::profiler::EventFactory;
using ::tensorflow::profiler::IOBufferAdapter;
using ::tensorflow::profiler::JsonTraceOptions;
using ::tensorflow::profiler::RawData;
using ::tensorflow::profiler::Trace;
using ::tensorflow::profiler::TraceEventArguments;
using ::tensorflow::profiler::TraceEventsContainerBase;
using ::tensorflow::profiler::ZstdCompression;

class TestEndToEndTraceContainer
    : public TraceEventsContainerBase<EventFactory, RawData> {
 public:
  explicit TestEndToEndTraceContainer(const Trace& trace) { trace_ = trace; }
  const Trace& trace() const { return trace_; }
  bool FilterByVisibility() const { return false; }
};

ParsedTraceEvents ParseJsonToParsedTraceEvents(absl::string_view json_str) {
  const nlohmann::json root = nlohmann::json::parse(json_str);
  ParsedTraceEvents result;
  if (root.contains("mpmdPipelineView") &&
      root["mpmdPipelineView"].is_boolean()) {
    result.mpmd_pipeline_view = root["mpmdPipelineView"].get<bool>();
  }
  if (root.contains("fullTimespan") && root["fullTimespan"].is_array() &&
      root["fullTimespan"].size() == 2) {
    result.full_timespan =
        std::make_pair(root["fullTimespan"][0].get<Milliseconds>(),
                       root["fullTimespan"][1].get<Milliseconds>());
  }
  if (!root.contains("traceEvents") || !root["traceEvents"].is_array()) {
    return result;
  }

  absl::flat_hash_map<std::pair<ProcessId, uint64_t>, TraceEvent>
      open_async_events;
  EventId next_event_id = 1;
  for (const nlohmann::json& event : root["traceEvents"]) {
    if (!event.contains("ph") || !event["ph"].is_string()) continue;
    const Phase ph = ParsePhase(event["ph"].get<std::string>());
    if (ph == Phase::kCounter) {
      if (!event.contains("entries")) continue;
      CounterEvent counter;
      if (event.contains("pid")) counter.pid = event["pid"].get<ProcessId>();
      if (event.contains("name")) {
        counter.name = event["name"].get<std::string>();
      }
      if (event.contains("event_stats")) {
        counter.event_stats = event["event_stats"].get<std::string>();
      }
      for (const nlohmann::json& entry : event["entries"]) {
        if (entry.is_array() && entry.size() == 2) {
          const Microseconds ts = entry[0].get<Microseconds>();
          const double val = entry[1].get<double>();
          counter.timestamps.push_back(ts);
          counter.values.push_back(val);
          counter.min_value = std::min(counter.min_value, val);
          counter.max_value = std::max(counter.max_value, val);
        }
      }
      result.counter_events.push_back(std::move(counter));
      continue;
    }

    TraceEvent ev;
    ev.ph = ph;
    ev.event_id = next_event_id++;
    if (event.contains("pid")) ev.pid = event["pid"].get<ProcessId>();
    if (event.contains("tid")) ev.tid = event["tid"].get<ThreadId>();
    if (event.contains("name")) ev.name = event["name"].get<std::string>();
    if (event.contains("ts")) ev.ts = event["ts"].get<Microseconds>();
    if (event.contains("dur")) {
      ev.dur = event["dur"].get<Microseconds>();
      // TraceEventsToJson clamps 0ps duration to 1ps (1e-6 us) for Catapult.
      if (std::abs(ev.dur - 1e-6) < 1e-12) {
        ev.dur = 0.0;
      }
    }
    if (event.contains("cat") && event["cat"].is_string()) {
      ev.category = GetContextTypeFromString(event["cat"].get<std::string>());
    }
    if (event.contains("id")) {
      if (event["id"].is_number_unsigned() || event["id"].is_number_integer()) {
        ev.flow_id = event["id"].get<uint64_t>();
        ev.has_flow_id = true;
      }
    } else if (event.contains("bind_id")) {
      if (event["bind_id"].is_number_unsigned() ||
          event["bind_id"].is_number_integer()) {
        ev.flow_id = event["bind_id"].get<uint64_t>();
        ev.has_flow_id = true;
      }
    }
    if (event.contains("args") && event["args"].is_object()) {
      for (const auto& [key, val] : event["args"].items()) {
        if (val.is_string()) {
          ev.args[key] = val.get<std::string>();
        } else if (val.is_number_integer()) {
          ev.args[key] = std::to_string(val.get<int64_t>());
        } else if (val.is_number_unsigned()) {
          ev.args[key] = std::to_string(val.get<uint64_t>());
        } else if (val.is_number_float()) {
          ev.args[key] = std::to_string(val.get<double>());
        } else if (val.is_boolean()) {
          ev.args[key] = val.get<bool>() ? "true" : "false";
        }
      }
    }
    if (event.contains("z") &&
        (event["z"].is_number_unsigned() || event["z"].is_number_integer())) {
      const uint64_t serial_val = event["z"].get<uint64_t>();
      ev.serial = serial_val;
      ev.has_serial = true;
      ev.args["uid"] = std::to_string(serial_val);
    }
    if (const auto gid_it = ev.args.find("group_id"); gid_it != ev.args.end()) {
      int64_t gid = 0;
      if (absl::SimpleAtoi(gid_it->second, &gid)) {
        ev.group_id = gid;
        ev.has_group_id = true;
      }
    }

    if (ev.ph == Phase::kAsyncBegin) {
      if (ev.flow_id != 0) {
        ev.is_async = true;
        open_async_events[{ev.pid, ev.flow_id}] = std::move(ev);
      }
    } else if (ev.ph == Phase::kAsyncEnd) {
      if (ev.flow_id != 0) {
        const auto it = open_async_events.find({ev.pid, ev.flow_id});
        if (it != open_async_events.end()) {
          TraceEvent& begin_ev = it->second;
          begin_ev.ph = Phase::kComplete;
          begin_ev.dur = ev.ts - begin_ev.ts;
          begin_ev.is_async = true;
          if (!ev.args.empty()) {
            begin_ev.args.insert(ev.args.begin(), ev.args.end());
          }
          result.flame_events.push_back(std::move(begin_ev));
          open_async_events.erase(it);
        }
      }
    } else {
      result.flame_events.push_back(std::move(ev));
    }
  }
  return result;
}

ParsedTraceEvents ParseProtoResponseToLegacyParsedTraceEvents(
    const xprof::TraceDataResponse& response) {
  ParsedTraceEvents result;
  for (const xprof::DetailItem& detail : response.details()) {
    if (detail.name() == tensorflow::profiler::kMpmdPipelineView) {
      result.mpmd_pipeline_view = detail.value();
    }
  }
  if (response.has_full_timespan_start_ps() &&
      response.has_full_timespan_end_ps()) {
    result.full_timespan =
        std::make_pair(response.full_timespan_start_ps() / 1e9,
                       response.full_timespan_end_ps() / 1e9);
  }
  ProcessMetadataEvents(response, result);
  ProcessCompleteEvents(response, result);
  ProcessAsyncEvents(response, result);
  ProcessCounterEvents(response, result);
  return result;
}

void AssertFlameChartTimelineDataParity(const FlameChartTimelineData& lhs,
                                        const FlameChartTimelineData& rhs,
                                        const DataProvider& dp_lhs,
                                        const DataProvider& dp_rhs,
                                        const Timeline& tl_lhs,
                                        const Timeline& tl_rhs) {
  ASSERT_EQ(lhs.groups.size(), rhs.groups.size());
  for (size_t i = 0; i < lhs.groups.size(); ++i) {
    SCOPED_TRACE(
        absl::StrCat("Group index ", i, " (", lhs.groups[i].name, ")"));
    EXPECT_EQ(lhs.groups[i].name, rhs.groups[i].name);
    EXPECT_EQ(lhs.groups[i].subtitle, rhs.groups[i].subtitle);
    EXPECT_EQ(lhs.groups[i].type, rhs.groups[i].type);
    EXPECT_EQ(lhs.groups[i].start_level, rhs.groups[i].start_level);
    EXPECT_EQ(lhs.groups[i].level_count, rhs.groups[i].level_count);
    EXPECT_EQ(lhs.groups[i].nesting_level, rhs.groups[i].nesting_level);
    EXPECT_EQ(lhs.groups[i].expanded, rhs.groups[i].expanded);
    EXPECT_EQ(lhs.groups[i].pid, rhs.groups[i].pid);
    EXPECT_EQ(lhs.groups[i].tid, rhs.groups[i].tid);
    EXPECT_EQ(lhs.groups[i].has_children, rhs.groups[i].has_children);
    EXPECT_EQ(lhs.groups[i].child_indices, rhs.groups[i].child_indices);
  }

  ASSERT_EQ(lhs.level_offsets, rhs.level_offsets);
  ASSERT_EQ(lhs.level_event_indices, rhs.level_event_indices);
  ASSERT_EQ(lhs.entry_levels, rhs.entry_levels);
  ASSERT_EQ(lhs.entry_start_times.size(), rhs.entry_start_times.size());
  ASSERT_EQ(lhs.entry_total_times.size(), rhs.entry_total_times.size());
  ASSERT_EQ(lhs.entry_name_indices.size(), rhs.entry_name_indices.size());

  for (size_t i = 0; i < lhs.entry_start_times.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Event index ", i));
    EXPECT_NEAR(lhs.entry_start_times[i], rhs.entry_start_times[i], 1e-6);
    EXPECT_NEAR(lhs.entry_total_times[i], rhs.entry_total_times[i], 1e-6);
    EXPECT_NEAR(lhs.GetEntrySelfTime(i), rhs.GetEntrySelfTime(i), 1e-6);

    const std::string& lhs_name =
        lhs.interned_string_pool[lhs.entry_name_indices[i]];
    const std::string& rhs_name =
        rhs.interned_string_pool[rhs.entry_name_indices[i]];
    EXPECT_EQ(lhs_name, rhs_name);

    EXPECT_EQ(lhs.GetEntryHloModule(i), rhs.GetEntryHloModule(i));
    EXPECT_EQ(lhs.GetEntryHloOp(i), rhs.GetEntryHloOp(i));
    const absl::flat_hash_map<std::string, std::string> lhs_args =
        lhs.GetEntryArgs(i);
    const absl::flat_hash_map<std::string, std::string> rhs_args =
        rhs.GetEntryArgs(i);
    if (lhs.entry_args.empty() && lhs.sparse_entry_args.empty() &&
        rhs.entry_args.empty() && rhs.sparse_entry_args.empty()) {
      EXPECT_EQ(lhs_args, rhs_args);
    } else {
      for (const absl::string_view key :
           {"uid", "group_id", "hlo_op", "hlo_module"}) {
        const auto lhs_it = lhs_args.find(key);
        const auto rhs_it = rhs_args.find(key);
        ASSERT_EQ(lhs_it != lhs_args.end(), rhs_it != rhs_args.end())
            << "Key mismatch: " << key;
        if (lhs_it != lhs_args.end()) {
          EXPECT_EQ(lhs_it->second, rhs_it->second) << "Key: " << key;
        }
      }
    }
    EXPECT_EQ(lhs.sparse_entry_group_ids.Find(i),
              rhs.sparse_entry_group_ids.Find(i));
  }
  EXPECT_EQ(lhs.sparse_entry_group_ids, rhs.sparse_entry_group_ids);

  ASSERT_EQ(lhs.flow_lines.size(), rhs.flow_lines.size());
  for (size_t i = 0; i < lhs.flow_lines.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("FlowLine index ", i));
    EXPECT_NEAR(lhs.flow_lines[i].source_ts, rhs.flow_lines[i].source_ts, 1e-6);
    EXPECT_NEAR(lhs.flow_lines[i].target_ts, rhs.flow_lines[i].target_ts, 1e-6);
    EXPECT_EQ(lhs.flow_lines[i].source_level, rhs.flow_lines[i].source_level);
    EXPECT_EQ(lhs.flow_lines[i].target_level, rhs.flow_lines[i].target_level);
    EXPECT_EQ(lhs.flow_lines[i].color, rhs.flow_lines[i].color);
    EXPECT_EQ(lhs.flow_lines[i].category, rhs.flow_lines[i].category);
  }

  ASSERT_EQ(lhs.counter_data_by_group_index.size(),
            rhs.counter_data_by_group_index.size());
  for (const auto& [group_idx, lhs_cnt] : lhs.counter_data_by_group_index) {
    const auto rhs_it = rhs.counter_data_by_group_index.find(group_idx);
    ASSERT_NE(rhs_it, rhs.counter_data_by_group_index.end());
    const CounterData& rhs_cnt = rhs_it->second;
    ASSERT_EQ(lhs_cnt.timestamps.size(), rhs_cnt.timestamps.size());
    for (size_t k = 0; k < lhs_cnt.timestamps.size(); ++k) {
      EXPECT_NEAR(lhs_cnt.timestamps[k], rhs_cnt.timestamps[k], 1e-6);
      EXPECT_NEAR(lhs_cnt.values[k], rhs_cnt.values[k], 1e-6);
    }
    EXPECT_NEAR(lhs_cnt.min_value, rhs_cnt.min_value, 1e-6);
    EXPECT_NEAR(lhs_cnt.max_value, rhs_cnt.max_value, 1e-6);
    EXPECT_EQ(lhs_cnt.event_stats, rhs_cnt.event_stats);
  }

  EXPECT_EQ(dp_lhs.GetProcessMappings(), dp_rhs.GetProcessMappings());
  EXPECT_EQ(dp_lhs.GetFlowCategories(), dp_rhs.GetFlowCategories());
  EXPECT_NEAR(tl_lhs.visible_range().start(), tl_rhs.visible_range().start(),
              1e-6);
  EXPECT_NEAR(tl_lhs.visible_range().end(), tl_rhs.visible_range().end(), 1e-6);
  EXPECT_NEAR(tl_lhs.fetched_data_time_range().start(),
              tl_rhs.fetched_data_time_range().start(), 1e-6);
  EXPECT_NEAR(tl_lhs.fetched_data_time_range().end(),
              tl_rhs.fetched_data_time_range().end(), 1e-6);
}

void AssertTimelineDataInvariants(const FlameChartTimelineData& data) {
  const size_t n = data.entry_start_times.size();
  // I1: Parallel SoA entry arrays have equal size and valid table indices.
  ASSERT_EQ(data.entry_total_times.size(), n);
  ASSERT_TRUE(data.entry_self_times.empty() ||
              data.entry_self_times.size() == n);
  ASSERT_EQ(data.entry_levels.size(), n);
  ASSERT_EQ(data.entry_name_indices.size(), n);
  ASSERT_TRUE(data.entry_hlo_op_indices.empty() ||
              data.entry_hlo_op_indices.size() == n);
  ASSERT_TRUE(data.entry_hlo_module_indices.empty() ||
              data.entry_hlo_module_indices.size() == n);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_LT(data.entry_name_indices[i], data.interned_string_pool.size());
    if (i < data.entry_hlo_op_indices.size()) {
      EXPECT_LT(data.entry_hlo_op_indices[i], data.hlo_op_table.size());
    } else if (const std::optional<uint32_t> explicit_op =
                   data.sparse_entry_hlo_op_indices.Find(i);
               explicit_op.has_value()) {
      EXPECT_LT(*explicit_op, data.hlo_op_table.size());
    }
    EXPECT_LT(data.GetEntryHloModuleIndex(i), data.hlo_module_table.size());
    // I4: 0 <= self_time <= total_time.
    const Microseconds self_time = data.GetEntrySelfTime(i);
    EXPECT_GE(self_time, -1e-6);
    EXPECT_LE(self_time, data.entry_total_times[i] + 1e-6);
  }

  // I2 & I3: CSR level_offsets and level_event_indices invariants.
  if (!data.level_offsets.empty()) {
    EXPECT_EQ(data.level_offsets.front(), 0u);
    EXPECT_EQ(data.level_offsets.back(), n);
    ASSERT_EQ(data.level_event_indices.size(), n);
    std::vector<bool> seen(n, false);
    for (size_t l = 0; l + 1 < data.level_offsets.size(); ++l) {
      ASSERT_LE(data.level_offsets[l], data.level_offsets[l + 1]);
      Microseconds prev_end = -std::numeric_limits<Microseconds>::infinity();
      for (size_t pos = data.level_offsets[l]; pos < data.level_offsets[l + 1];
           ++pos) {
        const size_t idx = data.level_event_indices[pos];
        ASSERT_LT(idx, n);
        EXPECT_FALSE(seen[idx]);
        seen[idx] = true;
        EXPECT_EQ(data.entry_levels[idx], l);
        EXPECT_GE(data.entry_start_times[idx] + 1e-6, prev_end);
        prev_end = data.entry_start_times[idx] + data.entry_total_times[idx];
      }
    }
  }

  // I6: Group hierarchy and counter invariants.
  for (size_t g = 0; g < data.groups.size(); ++g) {
    EXPECT_GE(data.groups[g].start_level, 0);
    EXPECT_GE(data.groups[g].level_count, 0);
    if (data.groups[g].parent_index != -1) {
      EXPECT_LT(static_cast<size_t>(data.groups[g].parent_index),
                data.groups.size());
    }
    if (data.groups[g].type == Group::Type::kCounter) {
      const auto it = data.counter_data_by_group_index.find(g);
      ASSERT_NE(it, data.counter_data_by_group_index.end());
      EXPECT_EQ(it->second.timestamps.size(), it->second.values.size());
      EXPECT_TRUE(absl::c_is_sorted(it->second.timestamps));
    }
  }

  // I7: GroupIdRuns half-open, sorted, disjoint, canonically coalesced runs.
  const absl::Span<const GroupIdRuns::Run> runs =
      data.sparse_entry_group_ids.runs();
  for (size_t k = 0; k < runs.size(); ++k) {
    EXPECT_LT(runs[k].begin, runs[k].end);
    EXPECT_LE(static_cast<size_t>(runs[k].end), n);
    if (k > 0) {
      EXPECT_LE(runs[k - 1].end, runs[k].begin);
      if (runs[k - 1].end == runs[k].begin) {
        EXPECT_NE(runs[k - 1].group_id, runs[k].group_id);
      }
    }
  }
}

void AssertTimelineDataSemanticEq(const FlameChartTimelineData& lhs,
                                  const FlameChartTimelineData& rhs,
                                  const DataProvider& dp_lhs,
                                  const DataProvider& dp_rhs,
                                  const Timeline& tl_lhs,
                                  const Timeline& tl_rhs) {
  AssertTimelineDataInvariants(lhs);
  AssertTimelineDataInvariants(rhs);
  AssertFlameChartTimelineDataParity(lhs, rhs, dp_lhs, dp_rhs, tl_lhs, tl_rhs);
}

class DataProviderParityTest : public ::testing::Test {
 protected:
  DataProviderParityTest() : palette_(ColorPalette::Default()) {}
  ColorPalette palette_;
};

// Parity across all 10 adversarial edge cases for both dense and sparse
// DeltaSeries protobuf modes, plus GetFlowCategories sorted determinism.
TEST_F(DataProviderParityTest,
       EndToEndJsonVsProtobufParityAcrossAdversarialEdgeCases) {
  Trace trace;
  trace.set_min_timestamp_ps(50ULL * 1000000ULL);
  trace.set_max_timestamp_ps(1000ULL * 1000000ULL);

  // Device 1: runs _layer_1 (so in MPMD view it sorts AFTER Device 2 which runs
  // _layer_0). Includes XLA Modules (tid 10), XLA Ops (tid 11), Compute
  // Utilization (tid 12), and a high-tid resource (tid 0x80000005).
  Device dev1;
  dev1.set_name("host0 /device:TPU:0");
  (*dev1.mutable_resources())[10].set_name(std::string(kXlaModules));
  (*dev1.mutable_resources())[11].set_name(std::string(kXlaOps));
  (*dev1.mutable_resources())[12].set_name(std::string(kComputeUtilization));
  (*dev1.mutable_resources())[0x80000005ULL].set_name("HighTidWorker");
  (*trace.mutable_devices())[1] = dev1;

  // Device 2: runs _layer_0 (sorts first in MPMD view).
  Device dev2;
  dev2.set_name("host0 /device:TPU:1");
  (*dev2.mutable_resources())[10].set_name(std::string(kXlaModules));
  (*dev2.mutable_resources())[11].set_name(std::string(kXlaOps));
  (*trace.mutable_devices())[2] = dev2;

  TestEndToEndTraceContainer container(trace);

  auto make_raw_args =
      [](std::initializer_list<std::pair<absl::string_view, absl::string_view>>
             kv_pairs) {
        RawData raw;
        for (const auto& [k, v] : kv_pairs) {
          TraceEventArguments::Argument* arg = raw.mutable_args()->add_arg();
          arg->set_name(std::string(k));
          arg->set_str_value(std::string(v));
        }
        return raw;
      };

  // Edge Case 2 & 3: Nested XLA Modules on Device 1 (tid 10):
  container.AddCompleteEvent("outer_mod(7)", /*resource_id=*/10,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 100ULL * 1000000ULL, 500ULL * 1000000ULL));
  RawData inner_mod_raw = make_raw_args({{"program_id", "8"}});
  container.AddCompleteEvent("inner_mod", /*resource_id=*/10, /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 200ULL * 1000000ULL, 300ULL * 1000000ULL),
                             &inner_mod_raw);

  // Edge Case 1, 4, 5: XLA Ops on Device 1 (tid 11):
  RawData custom_op_raw = make_raw_args(
      {{"hlo_op", "custom_parent_op"}, {"hlo_module", "custom_mod(99)"}});
  container.AddCompleteEvent("_layer_1.my_prog(1)", /*resource_id=*/11,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 100ULL * 1000000ULL, 250ULL * 1000000ULL),
                             &custom_op_raw, /*group_id=*/1, /*serial=*/501);

  container.AddCompleteEvent("child_fusion.1", /*resource_id=*/11,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 120ULL * 1000000ULL, 180ULL * 1000000ULL));

  container.AddCompleteEvent("partial_overlap_op", /*resource_id=*/11,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 200ULL * 1000000ULL, 320ULL * 1000000ULL));

  container.AddCompleteEvent("boundary_at_300us_op", /*resource_id=*/11,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 300ULL * 1000000ULL, 310ULL * 1000000ULL));

  container.AddCompleteEvent("post_inner_at_350us_op", /*resource_id=*/11,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 350ULL * 1000000ULL, 390ULL * 1000000ULL));

  // Edge Case 1: Compute Utilization (tid 12) event at 250us with "Name" arg
  RawData dml_raw = make_raw_args({{"Name", "dml_kernel_op"}});
  container.AddCompleteEvent("raw_dml_event", /*resource_id=*/12,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 250ULL * 1000000ULL, 270ULL * 1000000ULL),
                             &dml_raw);

  // Edge Case 6: Same-timestamp flow endpoints and multi-hop flow
  container.AddFlowEvent(
      "same_ts_flow_src", /*resource_id=*/11, /*device_id=*/1,
      tsl::profiler::Timespan::FromEndPoints(410ULL * 1000000ULL,
                                             415ULL * 1000000ULL),
      /*flow_id=*/9001, tensorflow::profiler::TraceEvent::FLOW_START,
      tsl::profiler::ContextType::kTfExecutor);
  container.AddFlowEvent(
      "same_ts_flow_dst", /*resource_id=*/12, /*device_id=*/1,
      tsl::profiler::Timespan::FromEndPoints(410ULL * 1000000ULL,
                                             415ULL * 1000000ULL),
      /*flow_id=*/9001, tensorflow::profiler::TraceEvent::FLOW_END,
      tsl::profiler::ContextType::kTfExecutor);

  container.AddFlowEvent("hop1_start", /*resource_id=*/11, /*device_id=*/1,
                         tsl::profiler::Timespan::FromEndPoints(
                             420ULL * 1000000ULL, 430ULL * 1000000ULL),
                         /*flow_id=*/9002,
                         tensorflow::profiler::TraceEvent::FLOW_START,
                         tsl::profiler::ContextType::kTpuLaunch);
  container.AddFlowEvent("hop2_mid", /*resource_id=*/12, /*device_id=*/1,
                         tsl::profiler::Timespan::FromEndPoints(
                             440ULL * 1000000ULL, 450ULL * 1000000ULL),
                         /*flow_id=*/9002,
                         tensorflow::profiler::TraceEvent::FLOW_MID,
                         tsl::profiler::ContextType::kTpuLaunch);
  container.AddFlowEvent("hop3_end", /*resource_id=*/11, /*device_id=*/2,
                         tsl::profiler::Timespan::FromEndPoints(
                             460ULL * 1000000ULL, 470ULL * 1000000ULL),
                         /*flow_id=*/9002,
                         tensorflow::profiler::TraceEvent::FLOW_END,
                         tsl::profiler::ContextType::kTpuLaunch);

  // Edge Case 7: High thread ID >= 0x80000000 (0x80000005)
  container.AddCompleteEvent("high_tid_event", /*resource_id=*/0x80000005ULL,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 150ULL * 1000000ULL, 190ULL * 1000000ULL));

  // Edge Case 8: Async event on Device 1 with hlo_op/hlo_module in RawData
  RawData async_raw =
      make_raw_args({{"hlo_op", "async_send_op"}, {"hlo_module", "async_mod"}});
  container.AddAsyncEvent(
      "MegaScale Channel", /*device_id=*/1,
      tsl::profiler::Timespan::FromEndPoints(130ULL * 1000000ULL, 0),
      /*flow_id=*/7001, tensorflow::profiler::TraceEvent::FLOW_START,
      tsl::profiler::ContextType::kGeneric, &async_raw);
  container.AddAsyncEvent(
      "MegaScale Channel", /*device_id=*/1,
      tsl::profiler::Timespan::FromEndPoints(230ULL * 1000000ULL, 0),
      /*flow_id=*/7001, tensorflow::profiler::TraceEvent::FLOW_END,
      tsl::profiler::ContextType::kGeneric);

  // Edge Case 9: Counter events (double + uint) on Device 1
  {
    RawData cnt_raw1;
    TraceEventArguments::Argument* a1 = cnt_raw1.mutable_args()->add_arg();
    a1->set_name("bytes");
    a1->set_double_value(3.14159);
    container.AddCounterEvent("HBM Bandwidth", /*device_id=*/1,
                              /*timestamp_ps=*/100ULL * 1000000ULL, cnt_raw1);

    RawData cnt_raw2;
    TraceEventArguments::Argument* a2 = cnt_raw2.mutable_args()->add_arg();
    a2->set_name("bytes");
    a2->set_uint_value(1048576ULL);
    container.AddCounterEvent("HBM Bandwidth", /*device_id=*/1,
                              /*timestamp_ps=*/200ULL * 1000000ULL, cnt_raw2);
  }

  // Edge Case 10: Device 2 XLA Module and _layer_0 XLA Op for MPMD sorting
  container.AddCompleteEvent("dev2_mod(1)", /*resource_id=*/10, /*device_id=*/2,
                             tsl::profiler::Timespan::FromEndPoints(
                                 80ULL * 1000000ULL, 600ULL * 1000000ULL));
  container.AddCompleteEvent("_layer_0.my_prog(1)", /*resource_id=*/11,
                             /*device_id=*/2,
                             tsl::profiler::Timespan::FromEndPoints(
                                 90ULL * 1000000ULL, 150ULL * 1000000ULL));

  // 1. Convert via JSON pipeline and ingest into Timeline/DataProvider.
  JsonTraceOptions json_options;
  json_options.mpmd_pipeline_view = true;
  std::string json_str;
  IOBufferAdapter json_output(&json_str);
  tensorflow::profiler::TraceEventsToJson<IOBufferAdapter,
                                          TestEndToEndTraceContainer, RawData>(
      json_options, container, &json_output);
  const ParsedTraceEvents parsed_json = ParseJsonToParsedTraceEvents(json_str);

  Timeline timeline_json(palette_);
  DataProvider dp_json;
  dp_json.ProcessTraceEvents(parsed_json, timeline_json);

  // 2. Convert via Compressed DeltaSeries Protobuf pipeline with
  // sparse_event_metadata = true (capable client) and ingest one-shot.
  DeltaSeriesProtoConversionOptions sparse_options;
  sparse_options.mpmd_pipeline_view = true;
  sparse_options.details = {{tensorflow::profiler::kMpmdPipelineView, true}};
  sparse_options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed_sparse,
      tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
          sparse_options, container));
  ASSERT_OK_AND_ASSIGN(const std::string decompressed_sparse,
                       ZstdCompression::Decompress(compressed_sparse));
  xprof::TraceDataResponse sparse_response;
  ASSERT_TRUE(sparse_response.ParseFromString(decompressed_sparse));

  Timeline timeline_proto_sparse(palette_);
  DataProvider dp_proto_sparse;
  dp_proto_sparse.ProcessTraceEvents(sparse_response, timeline_proto_sparse);

  // 3. Also convert with default dense options (sparse_event_metadata = false)
  // and verify dense vs sparse parity for non-HLO structural fields.
  DeltaSeriesProtoConversionOptions dense_options;
  dense_options.mpmd_pipeline_view = true;
  dense_options.details = {{tensorflow::profiler::kMpmdPipelineView, true}};
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed_dense,
      tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
          dense_options, container));
  ASSERT_OK_AND_ASSIGN(const std::string decompressed_dense,
                       ZstdCompression::Decompress(compressed_dense));
  xprof::TraceDataResponse dense_response;
  ASSERT_TRUE(dense_response.ParseFromString(decompressed_dense));

  Timeline timeline_proto_dense(palette_);
  DataProvider dp_proto_dense;
  dp_proto_dense.ProcessTraceEvents(dense_response, timeline_proto_dense);
  AssertTimelineDataInvariants(timeline_proto_dense.timeline_data());
  EXPECT_EQ(timeline_proto_dense.timeline_data().non_zero_uids_by_event_index,
            timeline_proto_sparse.timeline_data().non_zero_uids_by_event_index);
  EXPECT_EQ(timeline_proto_dense.timeline_data().sparse_entry_group_ids,
            timeline_proto_sparse.timeline_data().sparse_entry_group_ids);

  // 4. Verify full semantic parity between JSON and sparse Protobuf!
  AssertTimelineDataSemanticEq(
      timeline_json.timeline_data(), timeline_proto_sparse.timeline_data(),
      dp_json, dp_proto_sparse, timeline_json, timeline_proto_sparse);

  // Verify GetFlowCategories() is strictly sorted in ascending order.
  const std::vector<int> flow_cats = dp_proto_sparse.GetFlowCategories();
  EXPECT_EQ(flow_cats,
            (std::vector<int>{
                static_cast<int>(tsl::profiler::ContextType::kTfExecutor),
                static_cast<int>(tsl::profiler::ContextType::kTpuLaunch)}));
  EXPECT_TRUE(absl::c_is_sorted(flow_cats));
}

TEST_F(DataProviderParityTest, HloLookupGpuDmlTpuParity) {
  Trace trace;
  trace.set_min_timestamp_ps(0);
  trace.set_max_timestamp_ps(1000ULL * 1000000ULL);
  Device dev;
  dev.set_name("host0 /device:GPU:0");
  (*dev.mutable_resources())[1].set_name("Stream #0 (Compute)");
  (*dev.mutable_resources())[2].set_name(std::string(kComputeUtilization));
  (*dev.mutable_resources())[3].set_name(std::string(kXlaModules));
  (*dev.mutable_resources())[4].set_name(std::string(kXlaOps));
  (*trace.mutable_devices())[1] = dev;

  TestEndToEndTraceContainer container(trace);

  // 1. GPU kernel stream event with hlo_op, hlo_module, and kernel_details
  RawData gpu_kd_raw;
  {
    TraceEventArguments::Argument* a1 = gpu_kd_raw.mutable_args()->add_arg();
    a1->set_name("hlo_op");
    a1->set_str_value("gpu_gemm.1");
    TraceEventArguments::Argument* a2 = gpu_kd_raw.mutable_args()->add_arg();
    a2->set_name("hlo_module");
    a2->set_str_value("gpu_prog");
    TraceEventArguments::Argument* a3 = gpu_kd_raw.mutable_args()->add_arg();
    a3->set_name("kernel_details");
    a3->set_str_value("registers:64 module:gpu_prog_77 grid:128");
  }
  container.AddCompleteEvent("cutlass_gemm_kernel", /*resource_id=*/1,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 100ULL * 1000000ULL, 150ULL * 1000000ULL),
                             &gpu_kd_raw);

  // 2. GPU kernel stream event with hlo_op, hlo_module, and uint program_id
  RawData gpu_pid_raw;
  {
    TraceEventArguments::Argument* a1 = gpu_pid_raw.mutable_args()->add_arg();
    a1->set_name("hlo_op");
    a1->set_str_value("gpu_conv.2");
    TraceEventArguments::Argument* a2 = gpu_pid_raw.mutable_args()->add_arg();
    a2->set_name("hlo_module");
    a2->set_str_value("gpu_prog");
    TraceEventArguments::Argument* a3 = gpu_pid_raw.mutable_args()->add_arg();
    a3->set_name("program_id");
    a3->set_uint_value(88);
  }
  container.AddCompleteEvent("cudnn_conv_kernel", /*resource_id=*/1,
                             /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 160ULL * 1000000ULL, 200ULL * 1000000ULL),
                             &gpu_pid_raw);

  // 3. DML event with Name arg and XLA Module interval fallback
  container.AddCompleteEvent("tpu_mod(5)", /*resource_id=*/3, /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 50ULL * 1000000ULL, 900ULL * 1000000ULL));
  RawData dml_raw;
  {
    TraceEventArguments::Argument* a = dml_raw.mutable_args()->add_arg();
    a->set_name("Name");
    a->set_str_value("dml_copy.1");
  }
  container.AddCompleteEvent("dml_span", /*resource_id=*/2, /*device_id=*/1,
                             tsl::profiler::Timespan::FromEndPoints(
                                 220ULL * 1000000ULL, 260ULL * 1000000ULL),
                             &dml_raw);

  // 4. TPU XLA Ops track with > 64 events having program_id in raw_data (no
  // hlo_module) to exercise the O(1) events[0] probe and module interval scan.
  for (int i = 0; i < 70; ++i) {
    RawData tpu_raw;
    TraceEventArguments::Argument* a = tpu_raw.mutable_args()->add_arg();
    a->set_name("program_id");
    a->set_uint_value(5);
    const uint64_t start_ps = (300ULL + i * 5ULL) * 1000000ULL;
    container.AddCompleteEvent(
        absl::StrCat("tpu_fusion.", i), /*resource_id=*/4, /*device_id=*/1,
        tsl::profiler::Timespan::FromEndPoints(start_ps, start_ps + 2000000ULL),
        &tpu_raw);
  }

  std::string json_str;
  IOBufferAdapter json_output(&json_str);
  tensorflow::profiler::TraceEventsToJson<IOBufferAdapter,
                                          TestEndToEndTraceContainer, RawData>(
      JsonTraceOptions{}, container, &json_output);
  const ParsedTraceEvents parsed_json = ParseJsonToParsedTraceEvents(json_str);

  DeltaSeriesProtoConversionOptions proto_options;
  proto_options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed_proto,
      tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
          proto_options, container));
  ASSERT_OK_AND_ASSIGN(const std::string decompressed_proto,
                       ZstdCompression::Decompress(compressed_proto));
  xprof::TraceDataResponse proto_response;
  ASSERT_TRUE(proto_response.ParseFromString(decompressed_proto));

  // Verify O(1) probe suppressed sparse event_metadata on the 70-event TPU XLA
  // Ops track (resource 4).
  for (const xprof::TraceEventSeries& series :
       proto_response.complete_events()) {
    if (series.metadata().thread_id() == 4) {
      EXPECT_EQ(series.event_metadata_size(), 0);
    }
  }

  Timeline tl_j(palette_), tl_p_old(palette_), tl_p_new(palette_);
  DataProvider dp_j, dp_p_old, dp_p_new;
  dp_j.ProcessTraceEvents(parsed_json, tl_j);
  dp_p_old.ProcessTraceEvents(
      ParseProtoResponseToLegacyParsedTraceEvents(proto_response), tl_p_old);
  dp_p_new.ProcessTraceEvents(proto_response, tl_p_new);

  AssertTimelineDataSemanticEq(tl_j.timeline_data(), tl_p_new.timeline_data(),
                               dp_j, dp_p_new, tl_j, tl_p_new);
  AssertTimelineDataSemanticEq(tl_p_old.timeline_data(),
                               tl_p_new.timeline_data(), dp_p_old, dp_p_new,
                               tl_p_old, tl_p_new);
}

// 3-way differential oracle (AssertTimelineDataSemanticEq across P-new,
// P-old, and J) over 200 seeded random Trace instances (std::mt19937_64).
TEST_F(DataProviderParityTest,
       ThreeWayDifferentialOracle200SeededRandomTraces) {
  for (uint64_t seed = 1; seed <= 200; ++seed) {
    SCOPED_TRACE(absl::StrCat("Seed ", seed));
    std::mt19937_64 rng(seed);
    Trace trace;
    trace.set_min_timestamp_ps(10ULL * 1000000ULL);
    trace.set_max_timestamp_ps(2000ULL * 1000000ULL);

    const bool is_mpmd = (seed % 2 == 0);
    const bool use_sparse_metadata = (seed % 3 != 0);
    const uint32_t num_devices = 1 + static_cast<uint32_t>(rng() % 2);
    for (uint32_t pid = 1; pid <= num_devices; ++pid) {
      Device dev;
      dev.set_name(absl::StrCat("host", pid, " /device:TPU:", pid - 1));
      (*dev.mutable_resources())[10].set_name(std::string(kXlaModules));
      (*dev.mutable_resources())[11].set_name(std::string(kXlaOps));
      (*dev.mutable_resources())[12].set_name(std::string(kComputeUtilization));
      (*dev.mutable_resources())[13].set_name("Stream #0 (Compute)");
      (*trace.mutable_devices())[pid] = dev;
    }

    TestEndToEndTraceContainer container(trace);
    for (uint32_t pid = 1; pid <= num_devices; ++pid) {
      const uint64_t mod_start = (50 + (rng() % 50)) * 1000000ULL;
      const uint64_t mod_end = mod_start + (400 + (rng() % 300)) * 1000000ULL;
      container.AddCompleteEvent(
          absl::StrCat("mod_", pid, "(", seed % 7, ")"), /*resource_id=*/10,
          pid, tsl::profiler::Timespan::FromEndPoints(mod_start, mod_end));
      if (rng() % 2 == 0) {
        container.AddCompleteEvent(
            absl::StrCat("inner_mod_", pid, "(", (seed + 1) % 7, ")"),
            /*resource_id=*/10, pid,
            tsl::profiler::Timespan::FromEndPoints(mod_start + 50000000ULL,
                                                   mod_start + 150000000ULL));
      }

      const int num_ops = 3 + static_cast<int>(rng() % 5);
      uint64_t cur_ts = mod_start;
      for (int k = 0; k < num_ops; ++k) {
        cur_ts += (rng() % 40) * 1000000ULL;
        const uint64_t dur =
            (rng() % 4 == 0) ? 0 : (1 + (rng() % 60)) * 1000000ULL;
        const std::string op_name =
            (k == 0 && is_mpmd)
                ? absl::StrCat("_layer_", (num_devices - pid), ".op_", k)
                : absl::StrCat("fusion_", pid, "_", k);
        container.AddCompleteEvent(
            op_name, /*resource_id=*/11, pid,
            tsl::profiler::Timespan::FromEndPoints(cur_ts, cur_ts + dur),
            /*raw_data=*/nullptr, /*group_id=*/static_cast<int64_t>(k % 2 + 1),
            /*serial=*/static_cast<uint64_t>(100 * pid + k + 1));
      }

      if (use_sparse_metadata) {
        // DML event with Name arg
        RawData dml_raw;
        TraceEventArguments::Argument* dml_arg =
            dml_raw.mutable_args()->add_arg();
        dml_arg->set_name("Name");
        dml_arg->set_str_value(absl::StrCat("dml_", pid, "_", seed % 5));
        container.AddCompleteEvent(
            "dml_ev", /*resource_id=*/12, pid,
            tsl::profiler::Timespan::FromEndPoints(mod_start + 50000000ULL,
                                                   mod_start + 80000000ULL),
            &dml_raw);

        // GPU stream event with hlo_op + hlo_module + program_id
        RawData gpu_raw;
        TraceEventArguments::Argument* g_op = gpu_raw.mutable_args()->add_arg();
        g_op->set_name("hlo_op");
        g_op->set_str_value(absl::StrCat("gpu_op_", seed % 11));
        TraceEventArguments::Argument* g_mod =
            gpu_raw.mutable_args()->add_arg();
        g_mod->set_name("hlo_module");
        g_mod->set_str_value("gpu_m");
        TraceEventArguments::Argument* g_pid =
            gpu_raw.mutable_args()->add_arg();
        g_pid->set_name("program_id");
        g_pid->set_uint_value(seed % 13 + 1);
        container.AddCompleteEvent(
            "gpu_kernel", /*resource_id=*/13, pid,
            tsl::profiler::Timespan::FromEndPoints(mod_start + 20000000ULL,
                                                   mod_start + 60000000ULL),
            &gpu_raw);
      }

      // Async begin/end pair
      const uint64_t async_start = mod_start + (rng() % 100) * 1000000ULL;
      container.AddAsyncEvent(
          absl::StrCat("AsyncCh_", pid), pid,
          tsl::profiler::Timespan::FromEndPoints(async_start, 0),
          /*flow_id=*/5000 + pid, tensorflow::profiler::TraceEvent::FLOW_START);
      container.AddAsyncEvent(
          absl::StrCat("AsyncCh_", pid), pid,
          tsl::profiler::Timespan::FromEndPoints(async_start + 45000000ULL, 0),
          /*flow_id=*/5000 + pid, tensorflow::profiler::TraceEvent::FLOW_END);

      // Counter track
      RawData cnt_raw;
      cnt_raw.mutable_args()->add_arg()->set_double_value(
          static_cast<double>(seed % 100) + 0.5);
      container.AddCounterEvent("HBM", pid, mod_start + 10000000ULL, cnt_raw);
    }

    // Cross-thread flow
    container.AddFlowEvent("flow_src", /*resource_id=*/11, /*device_id=*/1,
                           tsl::profiler::Timespan::FromEndPoints(
                               700ULL * 1000000ULL, 720ULL * 1000000ULL),
                           /*flow_id=*/8800 + seed,
                           tensorflow::profiler::TraceEvent::FLOW_START,
                           tsl::profiler::ContextType::kTfExecutor);
    container.AddFlowEvent(
        "flow_dst", /*resource_id=*/12, /*device_id=*/num_devices,
        tsl::profiler::Timespan::FromEndPoints(730ULL * 1000000ULL,
                                               750ULL * 1000000ULL),
        /*flow_id=*/8800 + seed, tensorflow::profiler::TraceEvent::FLOW_END,
        tsl::profiler::ContextType::kTfExecutor);

    JsonTraceOptions json_opts;
    json_opts.mpmd_pipeline_view = is_mpmd;
    std::string json_str;
    IOBufferAdapter json_out(&json_str);
    tensorflow::profiler::TraceEventsToJson<
        IOBufferAdapter, TestEndToEndTraceContainer, RawData>(
        json_opts, container, &json_out);
    const ParsedTraceEvents parsed_j = ParseJsonToParsedTraceEvents(json_str);

    DeltaSeriesProtoConversionOptions proto_opts;
    proto_opts.mpmd_pipeline_view = is_mpmd;
    proto_opts.details = {{tensorflow::profiler::kMpmdPipelineView, is_mpmd}};
    proto_opts.sparse_event_metadata = use_sparse_metadata;
    ASSERT_OK_AND_ASSIGN(
        const std::string comp_pb,
        tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
            proto_opts, container));
    ASSERT_OK_AND_ASSIGN(const std::string decomp_pb,
                         ZstdCompression::Decompress(comp_pb));
    xprof::TraceDataResponse proto_resp;
    ASSERT_TRUE(proto_resp.ParseFromString(decomp_pb));
    const ParsedTraceEvents parsed_p_old =
        ParseProtoResponseToLegacyParsedTraceEvents(proto_resp);

    Timeline tl_j(palette_), tl_p_old(palette_), tl_p_new(palette_),
        tl_p_wire(palette_);
    DataProvider dp_j, dp_p_old, dp_p_new, dp_p_wire;
    dp_j.ProcessTraceEvents(parsed_j, tl_j);
    dp_p_old.ProcessTraceEvents(parsed_p_old, tl_p_old);
    dp_p_new.ProcessTraceEvents(proto_resp, tl_p_new);
    ASSERT_TRUE(dp_p_wire.ProcessSerializedTraceEvents(std::string(decomp_pb),
                                                       tl_p_wire));

    AssertTimelineDataSemanticEq(tl_j.timeline_data(), tl_p_new.timeline_data(),
                                 dp_j, dp_p_new, tl_j, tl_p_new);
    AssertTimelineDataSemanticEq(tl_p_old.timeline_data(),
                                 tl_p_new.timeline_data(), dp_p_old, dp_p_new,
                                 tl_p_old, tl_p_new);
    AssertTimelineDataSemanticEq(tl_p_new.timeline_data(),
                                 tl_p_wire.timeline_data(), dp_p_new, dp_p_wire,
                                 tl_p_new, tl_p_wire);
  }
}

// Complete 16-row event-taxonomy coverage matrix across J, P-old, and P-new
// with SCOPED_TRACE("Taxonomy row #k").
TEST_F(DataProviderParityTest, AllEventTaxonomy16RowCoverageMatrix) {
  for (int row = 1; row <= 16; ++row) {
    SCOPED_TRACE(absl::StrCat("Taxonomy row #", row));
    Trace trace;
    trace.set_min_timestamp_ps(0);
    trace.set_max_timestamp_ps(1000ULL * 1000000ULL);
    Device dev;
    dev.set_name("host0 /device:TPU:0");
    (*dev.mutable_resources())[10].set_name(std::string(kXlaModules));
    (*dev.mutable_resources())[11].set_name(std::string(kXlaOps));
    (*dev.mutable_resources())[12].set_name(std::string(kComputeUtilization));
    (*dev.mutable_resources())[13].set_name("Stream #0 (Compute)");
    (*dev.mutable_resources())[0x80000001ULL].set_name("HighTidSync");
    (*trace.mutable_devices())[1] = dev;

    TestEndToEndTraceContainer container(trace);
    switch (row) {
      case 1:  // Sync Complete event (dur > 0)
        container.AddCompleteEvent(
            "sync_complete", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   150ULL * 1000000ULL));
        break;
      case 2:  // Sync Instant event (dur == 0)
        container.AddCompleteEvent(
            "sync_instant", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   100ULL * 1000000ULL));
        break;
      case 3:  // Nested sync hierarchy (3 levels)
        container.AddCompleteEvent(
            "lvl0", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   300ULL * 1000000ULL));
        container.AddCompleteEvent(
            "lvl1", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(120ULL * 1000000ULL,
                                                   250ULL * 1000000ULL));
        container.AddCompleteEvent(
            "lvl2", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(150ULL * 1000000ULL,
                                                   200ULL * 1000000ULL));
        break;
      case 4:  // Partial overlap on same thread
        container.AddCompleteEvent(
            "p_ov1", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   200ULL * 1000000ULL));
        container.AddCompleteEvent(
            "p_ov2", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(150ULL * 1000000ULL,
                                                   250ULL * 1000000ULL));
        break;
      case 5:  // Tied (ts, dur) stability
        container.AddCompleteEvent(
            "tied_first", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   150ULL * 1000000ULL));
        container.AddCompleteEvent(
            "tied_second", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   150ULL * 1000000ULL));
        break;
      case 6:  // Nested XLA Modules + boundary op at start == inner.end
        container.AddCompleteEvent(
            "outer_m(1)", 10, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   500ULL * 1000000ULL));
        container.AddCompleteEvent(
            "inner_m(2)", 10, 1,
            tsl::profiler::Timespan::FromEndPoints(200ULL * 1000000ULL,
                                                   300ULL * 1000000ULL));
        container.AddCompleteEvent(
            "boundary_op", 11, 1,
            tsl::profiler::Timespan::FromEndPoints(300ULL * 1000000ULL,
                                                   320ULL * 1000000ULL));
        break;
      case 7: {  // GPU kernel stream hlo_op + hlo_module(program_id)
        RawData r;
        TraceEventArguments::Argument* a1 = r.mutable_args()->add_arg();
        a1->set_name("hlo_op");
        a1->set_str_value("gpu_op");
        TraceEventArguments::Argument* a2 = r.mutable_args()->add_arg();
        a2->set_name("hlo_module");
        a2->set_str_value("gpu_mod");
        TraceEventArguments::Argument* a3 = r.mutable_args()->add_arg();
        a3->set_name("program_id");
        a3->set_uint_value(9);
        container.AddCompleteEvent(
            "gpu_k", 13, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   140ULL * 1000000ULL),
            &r);
        break;
      }
      case 8: {  // GPU kernel stream hlo_module(kernel_details)
        RawData r;
        TraceEventArguments::Argument* a1 = r.mutable_args()->add_arg();
        a1->set_name("hlo_op");
        a1->set_str_value("gpu_kd_op");
        TraceEventArguments::Argument* a2 = r.mutable_args()->add_arg();
        a2->set_name("hlo_module");
        a2->set_str_value("kd_mod");
        TraceEventArguments::Argument* a3 = r.mutable_args()->add_arg();
        a3->set_name("kernel_details");
        a3->set_str_value("module:kd_mod_42");
        container.AddCompleteEvent(
            "gpu_kd", 13, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   140ULL * 1000000ULL),
            &r);
        break;
      }
      case 9: {  // DML Name arg
        RawData r;
        TraceEventArguments::Argument* a1 = r.mutable_args()->add_arg();
        a1->set_name("Name");
        a1->set_str_value("dml_op_name");
        container.AddCompleteEvent(
            "dml_ev", 12, 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL,
                                                   140ULL * 1000000ULL),
            &r);
        break;
      }
      case 10:  // Async FLOW_START + FLOW_END pair
        container.AddAsyncEvent(
            "AsyncPair", 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL, 0), 77,
            tensorflow::profiler::TraceEvent::FLOW_START);
        container.AddAsyncEvent(
            "AsyncPair", 1,
            tsl::profiler::Timespan::FromEndPoints(180ULL * 1000000ULL, 0), 77,
            tensorflow::profiler::TraceEvent::FLOW_END);
        break;
      case 11:  // Async FLOW_MID (emits start + end)
        container.AddAsyncEvent("AsyncMid", 1,
                                tsl::profiler::Timespan::FromEndPoints(
                                    100ULL * 1000000ULL, 160ULL * 1000000ULL),
                                88, tensorflow::profiler::TraceEvent::FLOW_MID);
        break;
      case 12:  // Same-timestamp flow endpoints
        container.AddFlowEvent("f_src", 11, 1,
                               tsl::profiler::Timespan::FromEndPoints(
                                   200ULL * 1000000ULL, 210ULL * 1000000ULL),
                               99, tensorflow::profiler::TraceEvent::FLOW_START,
                               tsl::profiler::ContextType::kTfExecutor);
        container.AddFlowEvent("f_dst", 12, 1,
                               tsl::profiler::Timespan::FromEndPoints(
                                   200ULL * 1000000ULL, 210ULL * 1000000ULL),
                               99, tensorflow::profiler::TraceEvent::FLOW_END,
                               tsl::profiler::ContextType::kTfExecutor);
        break;
      case 13:  // Multi-hop flow (FLOW_START -> FLOW_MID -> FLOW_END)
        container.AddFlowEvent("h1", 11, 1,
                               tsl::profiler::Timespan::FromEndPoints(
                                   100ULL * 1000000ULL, 110ULL * 1000000ULL),
                               101,
                               tensorflow::profiler::TraceEvent::FLOW_START,
                               tsl::profiler::ContextType::kTpuLaunch);
        container.AddFlowEvent("h2", 12, 1,
                               tsl::profiler::Timespan::FromEndPoints(
                                   120ULL * 1000000ULL, 130ULL * 1000000ULL),
                               101, tensorflow::profiler::TraceEvent::FLOW_MID,
                               tsl::profiler::ContextType::kTpuLaunch);
        container.AddFlowEvent("h3", 13, 1,
                               tsl::profiler::Timespan::FromEndPoints(
                                   140ULL * 1000000ULL, 150ULL * 1000000ULL),
                               101, tensorflow::profiler::TraceEvent::FLOW_END,
                               tsl::profiler::ContextType::kTpuLaunch);
        break;
      case 14: {  // Double counter track
        RawData c1, c2;
        c1.mutable_args()->add_arg()->set_double_value(1.25);
        c2.mutable_args()->add_arg()->set_double_value(8.75);
        container.AddCounterEvent("CntDbl", 1, 100ULL * 1000000ULL, c1);
        container.AddCounterEvent("CntDbl", 1, 200ULL * 1000000ULL, c2);
        break;
      }
      case 15: {  // Uint64 counter track
        RawData c1, c2;
        c1.mutable_args()->add_arg()->set_uint_value(1024);
        c2.mutable_args()->add_arg()->set_uint_value(4096);
        container.AddCounterEvent("CntUint", 1, 100ULL * 1000000ULL, c1);
        container.AddCounterEvent("CntUint", 1, 200ULL * 1000000ULL, c2);
        break;
      }
      case 16:  // Sync tid >= 0x80000000 under async process
        container.AddAsyncEvent(
            "AsyncProc", 1,
            tsl::profiler::Timespan::FromEndPoints(100ULL * 1000000ULL, 0), 202,
            tensorflow::profiler::TraceEvent::FLOW_START);
        container.AddAsyncEvent(
            "AsyncProc", 1,
            tsl::profiler::Timespan::FromEndPoints(150ULL * 1000000ULL, 0), 202,
            tensorflow::profiler::TraceEvent::FLOW_END);
        container.AddCompleteEvent(
            "high_tid_sync", 0x80000001ULL, 1,
            tsl::profiler::Timespan::FromEndPoints(110ULL * 1000000ULL,
                                                   140ULL * 1000000ULL));
        break;
    }

    std::string json_str;
    IOBufferAdapter json_out(&json_str);
    tensorflow::profiler::TraceEventsToJson<
        IOBufferAdapter, TestEndToEndTraceContainer, RawData>(
        JsonTraceOptions{}, container, &json_out);
    const ParsedTraceEvents parsed_j = ParseJsonToParsedTraceEvents(json_str);

    DeltaSeriesProtoConversionOptions proto_opts;
    proto_opts.sparse_event_metadata = true;
    ASSERT_OK_AND_ASSIGN(
        const std::string comp_pb,
        tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
            proto_opts, container));
    ASSERT_OK_AND_ASSIGN(const std::string decomp_pb,
                         ZstdCompression::Decompress(comp_pb));
    xprof::TraceDataResponse proto_resp;
    ASSERT_TRUE(proto_resp.ParseFromString(decomp_pb));
    const ParsedTraceEvents parsed_p_old =
        ParseProtoResponseToLegacyParsedTraceEvents(proto_resp);

    Timeline tl_j(palette_), tl_p_old(palette_), tl_p_new(palette_),
        tl_p_wire(palette_);
    DataProvider dp_j, dp_p_old, dp_p_new, dp_p_wire;
    dp_j.ProcessTraceEvents(parsed_j, tl_j);
    dp_p_old.ProcessTraceEvents(parsed_p_old, tl_p_old);
    dp_p_new.ProcessTraceEvents(proto_resp, tl_p_new);
    ASSERT_TRUE(dp_p_wire.ProcessSerializedTraceEvents(std::string(decomp_pb),
                                                       tl_p_wire));

    AssertTimelineDataSemanticEq(tl_j.timeline_data(), tl_p_new.timeline_data(),
                                 dp_j, dp_p_new, tl_j, tl_p_new);
    AssertTimelineDataSemanticEq(tl_p_old.timeline_data(),
                                 tl_p_new.timeline_data(), dp_p_old, dp_p_new,
                                 tl_p_old, tl_p_new);
    AssertTimelineDataSemanticEq(tl_p_new.timeline_data(),
                                 tl_p_wire.timeline_data(), dp_p_new, dp_p_wire,
                                 tl_p_new, tl_p_wire);
  }
}

// O3: Deterministic 200-seed streaming vs. monolithic parity oracle across
// random track segment splits and random byte chunk boundaries (1 B to 16 MiB).
TEST_F(DataProviderParityTest,
       StreamingTraceIngestorParity200SeededRandomTracesAndChunkSplits) {
  for (uint64_t seed = 1; seed <= 200; ++seed) {
    SCOPED_TRACE(absl::StrCat("O3 Seed ", seed));
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
    Trace trace;
    trace.set_min_timestamp_ps(10ULL * 1000000ULL);
    trace.set_max_timestamp_ps(2000ULL * 1000000ULL);

    const bool is_mpmd = (seed % 2 == 0);
    const bool use_sparse_metadata = (seed % 3 != 0);
    const uint32_t num_devices = 1 + static_cast<uint32_t>(rng() % 2);
    for (uint32_t pid = 1; pid <= num_devices; ++pid) {
      Device dev;
      dev.set_name(absl::StrCat("host", pid, " /device:TPU:", pid - 1));
      (*dev.mutable_resources())[10].set_name(std::string(kXlaModules));
      (*dev.mutable_resources())[11].set_name(std::string(kXlaOps));
      (*dev.mutable_resources())[12].set_name(std::string(kComputeUtilization));
      (*dev.mutable_resources())[13].set_name("Stream #0 (Compute)");
      (*trace.mutable_devices())[pid] = dev;
    }

    TestEndToEndTraceContainer container(trace);
    for (uint32_t pid = 1; pid <= num_devices; ++pid) {
      const uint64_t mod_start = (50 + (rng() % 50)) * 1000000ULL;
      const uint64_t mod_end = mod_start + (400 + (rng() % 300)) * 1000000ULL;
      container.AddCompleteEvent(
          absl::StrCat("mod_", pid, "(", seed % 7, ")"), /*resource_id=*/10,
          pid, tsl::profiler::Timespan::FromEndPoints(mod_start, mod_end));
      if (rng() % 2 == 0) {
        container.AddCompleteEvent(
            absl::StrCat("inner_mod_", pid, "(", (seed + 1) % 7, ")"),
            /*resource_id=*/10, pid,
            tsl::profiler::Timespan::FromEndPoints(mod_start + 50000000ULL,
                                                   mod_start + 150000000ULL));
      }

      const int num_ops = 4 + static_cast<int>(rng() % 6);
      uint64_t cur_ts = mod_start;
      for (int k = 0; k < num_ops; ++k) {
        cur_ts += (rng() % 30) * 1000000ULL;
        const uint64_t dur =
            (rng() % 5 == 0) ? 0 : (1 + (rng() % 50)) * 1000000ULL;
        const std::string op_name =
            (k == 0 && is_mpmd)
                ? absl::StrCat("_layer_", (num_devices - pid), ".op_", k)
                : absl::StrCat("fusion_", pid, "_", k);
        container.AddCompleteEvent(
            op_name, /*resource_id=*/11, pid,
            tsl::profiler::Timespan::FromEndPoints(cur_ts, cur_ts + dur),
            /*raw_data=*/nullptr, /*group_id=*/static_cast<int64_t>(k % 3 + 1),
            /*serial=*/static_cast<uint64_t>(100 * pid + k + 1));
      }

      // Paired async event with 64-bit flow_id (> 2^32).
      const uint64_t async_start = mod_start + (rng() % 80) * 1000000ULL;
      const uint64_t wide_async_flow_id =
          (uint64_t{1} << 35) | (seed * 17 + pid);
      container.AddAsyncEvent(
          absl::StrCat("AsyncCh_", pid), pid,
          tsl::profiler::Timespan::FromEndPoints(async_start, 0),
          /*flow_id=*/wide_async_flow_id,
          tensorflow::profiler::TraceEvent::FLOW_START);
      container.AddAsyncEvent(
          absl::StrCat("AsyncCh_", pid), pid,
          tsl::profiler::Timespan::FromEndPoints(async_start + 45000000ULL, 0),
          /*flow_id=*/wide_async_flow_id,
          tensorflow::profiler::TraceEvent::FLOW_END);

      // Unclosed async event on odd seeds to exercise P7 server-side trace_end
      // closing vs client-side closing.
      if (seed % 2 == 1) {
        container.AddAsyncEvent(absl::StrCat("UnclosedCh_", pid), pid,
                                tsl::profiler::Timespan::FromEndPoints(
                                    async_start + 10000000ULL, 0),
                                /*flow_id=*/wide_async_flow_id + 999,
                                tensorflow::profiler::TraceEvent::FLOW_START);
      }

      // Multiple counter points to exercise multi-segment counter merging.
      for (int c = 0; c < 3; ++c) {
        RawData cnt_raw;
        TraceEventArguments::Argument* cnt_arg =
            cnt_raw.mutable_args()->add_arg();
        cnt_arg->set_name("GiB");
        cnt_arg->set_double_value(static_cast<double>(seed % 100) + c * 1.25);
        container.AddCounterEvent("HBM", pid, mod_start + (c + 1) * 10000000ULL,
                                  cnt_raw);
      }
    }

    // Cross-thread flow with 64-bit flow_id.
    const uint64_t wide_flow_id = (uint64_t{1} << 36) | (8800 + seed);
    container.AddFlowEvent("flow_src", /*resource_id=*/11, /*device_id=*/1,
                           tsl::profiler::Timespan::FromEndPoints(
                               700ULL * 1000000ULL, 720ULL * 1000000ULL),
                           /*flow_id=*/wide_flow_id,
                           tensorflow::profiler::TraceEvent::FLOW_START,
                           tsl::profiler::ContextType::kTfExecutor);
    container.AddFlowEvent(
        "flow_dst", /*resource_id=*/12, /*device_id=*/num_devices,
        tsl::profiler::Timespan::FromEndPoints(730ULL * 1000000ULL,
                                               750ULL * 1000000ULL),
        /*flow_id=*/wide_flow_id, tensorflow::profiler::TraceEvent::FLOW_END,
        tsl::profiler::ContextType::kTfExecutor);

    DeltaSeriesProtoConversionOptions proto_opts;
    proto_opts.mpmd_pipeline_view = is_mpmd;
    proto_opts.details = {{tensorflow::profiler::kMpmdPipelineView, is_mpmd}};
    proto_opts.sparse_event_metadata = use_sparse_metadata;
    ASSERT_OK_AND_ASSIGN(
        const std::string comp_pb,
        tensorflow::profiler::ConvertTraceDataToCompressedDeltaSeriesProto(
            proto_opts, container));
    ASSERT_OK_AND_ASSIGN(const std::string decomp_pb,
                         ZstdCompression::Decompress(comp_pb));
    xprof::TraceDataResponse proto_resp;
    ASSERT_TRUE(proto_resp.ParseFromString(decomp_pb));

    // Encode into framed compressed stream with random per-track segment sizes
    // (1 event up to full remaining track size).
    ASSERT_OK_AND_ASSIGN(
        const std::string stream_bytes,
        tensorflow::profiler::ConvertTraceDataResponseToCompressedStreamProto(
            proto_resp,
            [&rng](uint32_t /*track_id*/, size_t remaining) -> size_t {
              if (remaining <= 1) return remaining;
              const size_t choice = static_cast<size_t>(rng() % 4);
              if (choice == 0) return 1;
              if (choice == 1) return std::min<size_t>(remaining, 2);
              if (choice == 2) {
                return 1 + static_cast<size_t>(rng() % remaining);
              }
              return remaining;
            }));

    Timeline tl_mono(palette_), tl_stream(palette_);
    DataProvider dp_mono, dp_stream;
    ASSERT_TRUE(
        dp_mono.ProcessSerializedTraceEvents(std::string(decomp_pb), tl_mono));

    StreamingTraceIngestor ingestor;
    ingestor.Begin(seed);
    size_t pos = 0;
    while (pos < stream_bytes.size()) {
      size_t chunk_len = 0;
      switch (rng() % 5) {
        case 0:
          chunk_len = 1;  // 1-byte stress boundary
          break;
        case 1:
          chunk_len = 2 + static_cast<size_t>(rng() % 16);
          break;
        case 2:
          chunk_len = 64 + static_cast<size_t>(rng() % 960);
          break;
        case 3:
          chunk_len = 4096 + static_cast<size_t>(rng() % 61440);
          break;
        default:
          chunk_len = 16U * 1024U * 1024U;  // 16 MiB upper bound
          break;
      }
      chunk_len = std::min(chunk_len, stream_bytes.size() - pos);
      ASSERT_TRUE(ingestor
                      .PushBytes(absl::string_view(stream_bytes.data() + pos,
                                                   chunk_len))
                      .ok());
      pos += chunk_len;
    }
    ASSERT_TRUE(ingestor.Finish(dp_stream, tl_stream).ok());

    AssertTimelineDataSemanticEq(tl_mono.timeline_data(),
                                 tl_stream.timeline_data(), dp_mono, dp_stream,
                                 tl_mono, tl_stream);
    // Bit-exact column equality check.
    EXPECT_EQ(tl_mono.timeline_data().entry_start_times,
              tl_stream.timeline_data().entry_start_times);
    EXPECT_EQ(tl_mono.timeline_data().entry_total_times,
              tl_stream.timeline_data().entry_total_times);
    EXPECT_EQ(tl_mono.timeline_data().entry_levels,
              tl_stream.timeline_data().entry_levels);
    EXPECT_EQ(tl_mono.timeline_data().level_offsets,
              tl_stream.timeline_data().level_offsets);
    EXPECT_EQ(tl_mono.timeline_data().level_event_indices,
              tl_stream.timeline_data().level_event_indices);
  }
}

// P10: Trailer count/checksum mismatch and mid-stream truncation discard
// staged state without mutating Timeline/DataProvider and recover cleanly via
// monolithic fallback.
TEST_F(DataProviderParityTest,
       StreamingTraceIngestorTrailerMismatchAndTruncationRecovery) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("KernelOp");
  response.set_full_timespan_start_ps(0);
  response.set_full_timespan_end_ps(10000000ULL);
  xprof::Process* proc = response.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("TPU 0");
  xprof::Thread* thr = proc->add_threads();
  thr->set_id(10);
  thr->set_name("XLA Ops");

  xprof::TraceEventSeries* s = response.add_complete_events();
  s->mutable_metadata()->set_process_id(1);
  s->mutable_metadata()->set_thread_id(10);
  s->add_deltas(1000000ULL);
  s->add_durations(500000ULL);
  s->add_name_refs(1);
  s->add_deltas(2000000ULL);
  s->add_durations(400000ULL);
  s->add_name_refs(1);

  ASSERT_OK_AND_ASSIGN(
      const std::string valid_stream,
      tensorflow::profiler::ConvertTraceDataResponseToCompressedStreamProto(
          response, /*max_events_per_segment=*/1));

  Timeline timeline(palette_);
  DataProvider data_provider;

  // 1. Truncated stream (missing trailer / cut mid-frame) must fail Finish and
  // leave timeline empty.
  StreamingTraceIngestor ingestor;
  ingestor.Begin(1);
  ASSERT_GT(valid_stream.size(), 10u);
  EXPECT_TRUE(ingestor
                  .PushBytes(absl::string_view(valid_stream.data(),
                                               valid_stream.size() - 6))
                  .ok());
  EXPECT_FALSE(ingestor.Finish(data_provider, timeline).ok());
  EXPECT_TRUE(timeline.timeline_data().entry_start_times.empty());

  // 2. Fallback to monolithic serialized response succeeds cleanly.
  std::string serialized_monolithic;
  ASSERT_TRUE(response.SerializeToString(&serialized_monolithic));
  ASSERT_TRUE(data_provider.ProcessSerializedTraceEvents(
      std::move(serialized_monolithic), timeline));
  ASSERT_EQ(timeline.timeline_data().entry_start_times.size(), 2u);
}

}  // namespace
}  // namespace traceviewer
