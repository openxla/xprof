#include "xprof/convert/megascale_perfetto/trace_processor.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/strings/string_view.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {
namespace {

TEST(TraceProcessorTest, SortsEvents) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA Ops";

  // Unsorted events
  track.events.push_back(
      {"op 1", /*ts=*/100,
       /*duration=*/50});  // Below 1ns is ok (only 1 event, won't group)
  track.events.push_back({"op 2", /*ts=*/200, /*duration=*/1500});
  // If timestamps are the same, longer duration should come first.
  track.events.push_back({"op 3", /*ts=*/200, /*duration=*/2000});
  track.events.push_back({"op 4", /*ts=*/50, /*duration=*/1200});

  TraceProcessor processor(&trace);
  processor.Process();

  ASSERT_EQ(track.events.size(), 4);
  EXPECT_EQ(track.events[0].name, "op 4");
  EXPECT_EQ(track.events[1].name, "op 1");
  EXPECT_EQ(track.events[2].name, "op 3");
  EXPECT_EQ(track.events[3].name, "op 2");
}

TEST(TraceProcessorTest, AssignsRunIds) {
  XprofTrace trace;
  auto& tpu_tracks = trace.tpu_fragments[0];
  tpu_tracks.reserve(2);  // Prevent reallocation from invalidating references

  Track& modules = tpu_tracks.emplace_back();
  modules.name = "XLA Modules";

  // Add two runs. The first one already has a run_id. The second one
  // doesn't, so it should be assigned the next available run_id.
  Event m1{"module 1", /*ts=*/100, /*duration=*/1000};
  m1.args.push_back({trace.string_table.Intern("run_id"), uint64_t{123}});
  modules.events.push_back(std::move(m1));
  modules.events.push_back({"module 2", /*ts=*/2000, /*duration=*/1000});

  Track& ops = tpu_tracks.emplace_back();
  ops.name = "XLA Ops";
  ops.events.push_back({"op 1.1", /*ts=*/150, /*duration=*/10});
  ops.events.push_back({"op 2.1", /*ts=*/2100, /*duration=*/10});

  Track& ms0 = trace.megascale_fragments[0].emplace_back();
  ms0.name = "rendezvous";
  ms0.events.push_back({"DeviceToHost END", /*ts=*/160, /*duration=*/10});

  TraceProcessor processor(&trace);
  processor.Process();

  EXPECT_EQ(modules.events[0].run_id, 123);
  EXPECT_EQ(modules.events[1].run_id, 1);
  EXPECT_EQ(ops.events[0].run_id, 123);
  EXPECT_EQ(ops.events[1].run_id, 1);
  EXPECT_EQ(ms0.events[0].run_id, 123);
}

TEST(TraceProcessorTest, MarksLastDmaEvents) {
  XprofTrace trace;
  Track& track = trace.megascale_fragments[0].emplace_back();
  track.name = "rendezvous";

  // Execution event
  track.events.push_back(
      {"device_0_gid_rendezvous_0", /*ts=*/100, /*duration=*/1000});
  track.events.at(0).run_id = 1;

  // DMA events within execution
  track.events.push_back({"DeviceToHost END", /*ts=*/200, /*duration=*/10});
  track.events.back().run_id = 1;
  track.events.push_back({"DeviceToHost END", /*ts=*/300, /*duration=*/10});
  track.events.back().run_id = 1;
  track.events.push_back({"HostToDevice END", /*ts=*/400, /*duration=*/10});
  track.events.back().run_id = 1;

  TraceProcessor processor(&trace);
  processor.Process();

  auto find_is_last = [&](const Event& event) {
    for (const auto& arg : event.args) {
      if (trace.string_table.Get(arg.key) == "is_last_instance") {
        return std::get<int64_t>(arg.value) == 1;
      }
    }
    return false;
  };

  EXPECT_FALSE(find_is_last(track.events[1]));
  EXPECT_TRUE(find_is_last(track.events[2]));  // Last D2H
  EXPECT_TRUE(find_is_last(track.events[3]));  // Last H2D
}

TEST(TraceProcessorTest, AddsNetworkCounters) {
  XprofTrace trace;
  Track& track = trace.megascale_fragments[0].emplace_back();
  track.name = "rendezvous";

  Event event;
  event.name = "NetworkReceive END";
  event.timestamp_ps = 2000000;  // 2us
  event.duration_ps = 1000000;   // 1us
  event.run_id = 1;
  event.args.push_back(
      {trace.string_table.Intern("network_transport_latency_us"), uint64_t{1}});
  event.args.push_back(
      {trace.string_table.Intern("buffer_sizes"),
       trace.string_table.Intern("s100001->200001d|$c0=1000")});
  track.events.push_back(std::move(event));

  TraceProcessor processor(&trace);
  processor.Process();

  // Check outstanding bytes counter.
  EXPECT_FALSE(trace.rx_counter.values.empty());
  EXPECT_EQ(trace.rx_counter.name, "Outstanding Bytes RX");
  // Start: +1000 bytes at 2us
  // End: -1000 bytes at 3us
  ASSERT_EQ(trace.rx_counter.timestamps_ps.size(), 2);
  EXPECT_EQ(trace.rx_counter.timestamps_ps[0], 2000000);
  EXPECT_EQ(trace.rx_counter.values[0], 1000);
  EXPECT_EQ(trace.rx_counter.timestamps_ps[1], 3000000);
  EXPECT_EQ(trace.rx_counter.values[1], 0);

  // Check network bandwidth counter.
  EXPECT_FALSE(trace.rx_bw_counter.values.empty());
  EXPECT_EQ(trace.rx_bw_counter.name, "Bandwidth RX (Gbps)");
  // Start: +8 Gbps at 2us
  // End: -8 Gbps at 3us
  ASSERT_EQ(trace.rx_bw_counter.timestamps_ps.size(), 2);
  EXPECT_EQ(trace.rx_bw_counter.timestamps_ps[0], 2000000);
  EXPECT_EQ(trace.rx_bw_counter.values[0], 8);
  EXPECT_EQ(trace.rx_bw_counter.timestamps_ps[1], 3000000);
  EXPECT_EQ(trace.rx_bw_counter.values[1], 0);
}

TEST(TraceProcessorTest, AddMegascaleCounters) {
  XprofTrace trace;
  Track& track = trace.megascale_fragments[0].emplace_back();
  track.name = "rendezvous";

  Event event;
  event.name = "device_0_gid_rendezvous_0";
  event.timestamp_ps = 2000000;  // 2us
  event.duration_ps = 1000000;   // 1us
  event.run_id = 1;
  event.args.push_back(
      {trace.string_table.Intern("input_size"), int64_t{1000}});
  track.events.push_back(std::move(event));

  TraceProcessor processor(&trace);
  processor.Process();

  EXPECT_FALSE(trace.inflight_collective_count.values.empty());
  EXPECT_EQ(trace.inflight_collective_count.name, "Inflight Collectives");
  ASSERT_EQ(trace.inflight_collective_count.timestamps_ps.size(), 2);
  EXPECT_EQ(trace.inflight_collective_count.timestamps_ps[0], 2000000);
  EXPECT_EQ(trace.inflight_collective_count.values[0], 1);
  EXPECT_EQ(trace.inflight_collective_count.timestamps_ps[1], 3000000);
  EXPECT_EQ(trace.inflight_collective_count.values[1], 0);

  EXPECT_FALSE(trace.inflight_collective_bytes.values.empty());
  EXPECT_EQ(trace.inflight_collective_bytes.name,
            "Inflight Collective Payload");
  ASSERT_EQ(trace.inflight_collective_bytes.timestamps_ps.size(), 2);
  EXPECT_EQ(trace.inflight_collective_bytes.timestamps_ps[0], 2000000);
  EXPECT_EQ(trace.inflight_collective_bytes.values[0], 1000);
  EXPECT_EQ(trace.inflight_collective_bytes.timestamps_ps[1], 3000000);
  EXPECT_EQ(trace.inflight_collective_bytes.values[1], 0);
}

TEST(TraceProcessorTest, ModifiesTrackNames) {
  XprofTrace trace;
  trace.tpu_fragments[0].emplace_back(Track{"Steps", {}});
  trace.megascale_fragments[0].emplace_back(
      Track{"device_0_gid_rendezvous_0", {}});

  TraceProcessor processor(&trace);
  processor.Process();

  EXPECT_EQ(trace.tpu_fragments[0][0].name, "1. Steps");
  // The name is NOT modified because RenameTrack fails to extract groups
  // from kGraphNameRe (which now has only 1 group).
  EXPECT_EQ(trace.megascale_fragments[0][0].name, "device_0_gid_rendezvous_0");
}

TEST(TraceProcessorTest, ResolvesFlows) {
  XprofTrace trace;
  auto& tpu_tracks = trace.tpu_fragments[0];
  tpu_tracks.reserve(2);

  Track& modules = tpu_tracks.emplace_back();
  modules.name = "XLA Modules";
  Event m1{"module 1", /*ts=*/50, /*duration=*/1000};
  m1.args.push_back({trace.string_table.Intern("run_id"), uint64_t{123}});
  modules.events.push_back(std::move(m1));

  Track& ops = tpu_tracks.emplace_back();
  ops.name = "XLA Ops";
  ops.events.reserve(5);

  Event send_event{"send", /*ts=*/150, /*duration=*/10};
  send_event.args.push_back({trace.string_table.Intern("long_name"),
                             trace.string_table.Intern("channel_id=42")});
  ops.events.push_back(std::move(send_event));
  Event& send_ref = ops.events.back();

  Event recv_event{"recv", /*ts=*/160, /*duration=*/10};
  recv_event.args.push_back({trace.string_table.Intern("long_name"),
                             trace.string_table.Intern("channel_id=43")});
  ops.events.push_back(std::move(recv_event));
  Event& recv_ref = ops.events.back();

  Event send_done_event{"send-done", /*ts=*/400, /*duration=*/10};
  send_done_event.args.push_back({trace.string_table.Intern("long_name"),
                                  trace.string_table.Intern("channel_id=42")});
  ops.events.push_back(std::move(send_done_event));
  Event& send_done_ref = ops.events.back();

  Event recv_done_event{"recv-done", /*ts=*/500, /*duration=*/10};
  recv_done_event.args.push_back({trace.string_table.Intern("long_name"),
                                  trace.string_table.Intern("channel_id=43")});
  ops.events.push_back(std::move(recv_done_event));
  Event& recv_done_ref = ops.events.back();

  Track& ms = trace.megascale_fragments[0].emplace_back();
  ms.name = "rendezvous";
  ms.events.reserve(5);

  Event header_event{"device_0_gid_rendezvous_0", /*ts=*/50, /*duration=*/2000};
  header_event.args.push_back({trace.string_table.Intern("send_channel_id"),
                               trace.string_table.Intern("42")});
  header_event.args.push_back({trace.string_table.Intern("recv_channel_id"),
                               trace.string_table.Intern("43")});
  ms.events.push_back(std::move(header_event));

  Event d2h_start{"DeviceToHost START", /*ts=*/200, /*duration=*/10};
  d2h_start.args.push_back(
      {trace.string_table.Intern("graph_key"),
       trace.string_table.Intern("device_0_gid_rendezvous_0")});
  d2h_start.args.push_back(
      {trace.string_table.Intern("action_index"), int64_t{0}});
  ms.events.push_back(std::move(d2h_start));
  Event& d2h_start_ref = ms.events.back();

  Event d2h_end{"DeviceToHost END", /*ts=*/250, /*duration=*/10};
  d2h_end.args.push_back(
      {trace.string_table.Intern("graph_key"),
       trace.string_table.Intern("device_0_gid_rendezvous_0")});
  d2h_end.args.push_back(
      {trace.string_table.Intern("is_last_instance"), int64_t{1}});
  ms.events.push_back(std::move(d2h_end));
  Event& d2h_end_ref = ms.events.back();

  Event h2d_start{"HostToDevice START", /*ts=*/300, /*duration=*/10};
  h2d_start.args.push_back(
      {trace.string_table.Intern("graph_key"),
       trace.string_table.Intern("device_0_gid_rendezvous_0")});
  h2d_start.args.push_back(
      {trace.string_table.Intern("action_index"), int64_t{0}});
  ms.events.push_back(std::move(h2d_start));
  Event& h2d_start_ref = ms.events.back();

  Event h2d_end{"HostToDevice END", /*ts=*/350, /*duration=*/10};
  h2d_end.args.push_back(
      {trace.string_table.Intern("graph_key"),
       trace.string_table.Intern("device_0_gid_rendezvous_0")});
  h2d_end.args.push_back(
      {trace.string_table.Intern("is_last_instance"), int64_t{1}});
  ms.events.push_back(std::move(h2d_end));
  Event& h2d_end_ref = ms.events.back();

  TraceProcessor processor(&trace);
  processor.Process();

  auto get_flow_ids = [&](const Event& event, FlowDirection dir) {
    std::vector<int64_t> ids;
    for (const auto& flow : event.flows) {
      if (flow.direction == dir) {
        ids.push_back(flow.id);
      }
    }
    return ids;
  };

  auto has_common_element = [](const std::vector<int64_t>& a,
                               const std::vector<int64_t>& b) {
    for (auto x : a) {
      for (auto y : b) {
        if (x == y) return true;
      }
    }
    return false;
  };

  // 1. send -> send-done
  std::vector<int64_t> send_out =
      get_flow_ids(send_ref, FlowDirection::kSource);
  std::vector<int64_t> send_done_in =
      get_flow_ids(send_done_ref, FlowDirection::kSink);
  EXPECT_TRUE(has_common_element(send_out, send_done_in));

  // 2. send -> recv-done
  std::vector<int64_t> recv_done_in =
      get_flow_ids(recv_done_ref, FlowDirection::kSink);
  EXPECT_TRUE(has_common_element(send_out, recv_done_in));

  // 3. send -> D2H start
  std::vector<int64_t> d2h_start_in =
      get_flow_ids(d2h_start_ref, FlowDirection::kSink);
  EXPECT_TRUE(has_common_element(send_out, d2h_start_in));

  // 4. recv -> recv-done
  std::vector<int64_t> recv_out =
      get_flow_ids(recv_ref, FlowDirection::kSource);
  EXPECT_TRUE(has_common_element(recv_out, recv_done_in));

  // 5. recv -> H2D start
  std::vector<int64_t> h2d_start_in =
      get_flow_ids(h2d_start_ref, FlowDirection::kSink);
  EXPECT_TRUE(has_common_element(recv_out, h2d_start_in));

  // 6. D2H END -> send-done
  std::vector<int64_t> d2h_end_out =
      get_flow_ids(d2h_end_ref, FlowDirection::kSource);
  EXPECT_TRUE(has_common_element(d2h_end_out, send_done_in));

  // 7. H2D END -> recv-done (recv-done END)
  std::vector<int64_t> h2d_end_out =
      get_flow_ids(h2d_end_ref, FlowDirection::kSource);

  // Verify that a recv-done END event is added and that it has the expected
  // inflows.
  ASSERT_EQ(ops.events.size(), 5);
  EXPECT_EQ(ops.events[4].name, "recv-done END");
  Event& recv_done_end_ref = ops.events[4];
  std::vector<int64_t> recv_done_end_in =
      get_flow_ids(recv_done_end_ref, FlowDirection::kSink);
  EXPECT_TRUE(has_common_element(h2d_end_out, recv_done_end_in));
}

TEST(TraceProcessorTest, GroupsTinyEvents) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA Ops";

  // Tiny events at the same truncated nanosecond (100ns -> 100,000ps)
  // ts=100.0ns, dur=0.2ns, run_id=-1
  track.events.push_back({"Fusion:op1", 100000, 200});
  // ts=100.3ns, dur=0.1ns, run_id=-1
  track.events.push_back({"Fusion:op2", 100300, 100});
  // ts=100.5ns, dur=0.3ns, run_id=-1
  track.events.push_back({"Fusion:op3", 100500, 300});

  // Non-colliding tiny event at another timestamp (201ns -> 201,000ps)
  // ts=201.0ns, dur=0.1ns, run_id=-1
  track.events.push_back({"Fusion:op4", 201000, 100});

  // Large compute event at the first timestamp (dur >= 1ns [1000ps])
  // ts=100.8ns, dur=1.5ns, run_id=-1
  track.events.push_back({"LargeOp", 100800, 1500});

  // Isolated tiny event mapping to a different nanosecond (ts/1000 = 8)
  // ts=800.0ns, dur=0.1ns, run_id=-1
  track.events.push_back({"Fusion:isolated", 800000, 100});

  TraceProcessor processor(&trace, /*group_tiny_events=*/true);
  processor.Process();

  // Ops 1, 2, and 3 should be grouped in-place.
  // LargeOp has duration >= 1ns, so remains raw.
  // Fusion:op4 is tiny but stands raw since it is alone at 201ns.
  // Expected track layout order after processing and sorting:
  // 1. SummaryBlock: ts=100,000ps, dur=800ps
  // 2. LargeOp: ts=100,800ps, dur=1500ps
  // 3. Fusion:op4: ts=201,000ps, dur=100ps
  // 4. Fusion:isolated: ts=800,000ps, dur=100ps
  ASSERT_EQ(track.events.size(), 4);

  EXPECT_EQ(track.events[0].name, "3 events are hidden");
  EXPECT_EQ(track.events[0].timestamp_ps, 100000);
  EXPECT_EQ(track.events[0].duration_ps, 800);

  auto get_arg_str = [&](const Event& ev,
                         absl::string_view key) -> std::string {
    for (const auto& arg : ev.args) {
      if (trace.string_table.Get(arg.key) == key) {
        return std::string(
            trace.string_table.Get(std::get<StringId>(arg.value)));
      }
    }
    return "";
  };

  // Wrapped string literals complying with 80-character maximum boundaries
  EXPECT_EQ(get_arg_str(track.events[0], "description"),
            "If you would like to see them, please add "
            "&group_tiny_events=false to the URL.");
  EXPECT_EQ(get_arg_str(track.events[0], "hidden_events"),
            "Fusion:op1, Fusion:op2, Fusion:op3");

  // Verify that redundant run_id is NOT in the args list!
  bool has_run_id_arg = false;
  for (const auto& arg : track.events[0].args) {
    if (trace.string_table.Get(arg.key) == "run_id") {
      has_run_id_arg = true;
    }
  }
  EXPECT_FALSE(has_run_id_arg);

  EXPECT_EQ(track.events[1].name, "LargeOp");
  EXPECT_EQ(track.events[2].name, "Fusion:op4");
  EXPECT_EQ(track.events[3].name, "Fusion:isolated");
}

TEST(TraceProcessorTest,
     ExemptSyncPrimitiveBreaksGroupingAndPreservesFlatLayout) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA Ops";

  // Intermixed tiny compute and sync primitives at the same truncated
  // timestamp:
  // ts=100.0ns, dur=0.2ns (tiny)
  track.events.push_back({"Fusion:op1", 100000, 200});
  // ts=100.3ns, dur=0.3ns (sync, exempt)
  track.events.push_back({"send.3", 100300, 300});
  // ts=100.7ns, dur=0.1ns (tiny)
  track.events.push_back({"Fusion:op2", 100700, 100});

  TraceProcessor processor(&trace, /*group_tiny_events=*/true);
  processor.Process();

  // Under "In-Place Contiguous Break" strategy, the sync primitive "send.3"
  // breaks grouping contiguity instantly. Active group [op1] is flushed
  // immediately (size=1 -> raw), "send.3" is pushed raw, and the new group
  // [op2] is flushed raw at the end (size=1 -> raw). Verification: Grouping is
  // entirely bypassed because no contiguous sequences of eligible tiny events
  // reach the trigger threshold (N >= 2). All three slices remain raw and flat
  // on the timeline.
  ASSERT_EQ(track.events.size(), 3);

  EXPECT_EQ(track.events[0].name, "Fusion:op1");
  EXPECT_EQ(track.events[0].timestamp_ps, 100000);
  EXPECT_EQ(track.events[0].duration_ps, 200);

  EXPECT_EQ(track.events[1].name, "send.3");
  EXPECT_EQ(track.events[1].timestamp_ps, 100300);
  EXPECT_EQ(track.events[1].duration_ps, 300);

  EXPECT_EQ(track.events[2].name, "Fusion:op2");
  EXPECT_EQ(track.events[2].timestamp_ps, 100700);
  EXPECT_EQ(track.events[2].duration_ps, 100);
}

TEST(TraceProcessorTest, GroupingEnforcesRunIdMatch) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA Ops";

  // Adjacent tiny events under the same truncated nanosecond
  // (100ns -> 100,000ps), but residing in completely separate execution
  // program run boundaries (run_id 1 vs run_id 2).
  Event e1{"Fusion:op1", 100000, 200};  // ts=100.0ns
  e1.run_id = 1;
  track.events.push_back(std::move(e1));

  Event e2{"Fusion:op2", 100400, 200};  // ts=100.4ns
  e2.run_id = 2;
  track.events.push_back(std::move(e2));

  TraceProcessor processor(&trace, /*group_tiny_events=*/true);
  processor.Process();

  // Folding tiny events across run boundaries must be prevented. Slices
  // remain raw!
  ASSERT_EQ(track.events.size(), 2);
  EXPECT_EQ(track.events[0].name, "Fusion:op1");
  EXPECT_EQ(track.events[1].name, "Fusion:op2");
}

TEST(TraceProcessorTest, GroupingToggleDisablesPass) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA Ops";

  track.events.push_back({"Fusion:op1", 100000, 200});
  track.events.push_back({"Fusion:op2", 100300, 100});

  // group_tiny_events = false -> Disables summary grouping layout
  // optimization!
  TraceProcessor processor(&trace, /*group_tiny_events=*/false);
  processor.Process();

  ASSERT_EQ(track.events.size(), 2);
  EXPECT_EQ(track.events[0].name, "Fusion:op1");
  EXPECT_EQ(track.events[1].name, "Fusion:op2");
}

TEST(TraceProcessorTest, GroupsTinyEventsOnTraceMe) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA TraceMe";

  // Tiny events at the same truncated nanosecond (100ns -> 100,000ps)
  track.events.push_back({"TraceMe:op1", 100000, 200});
  track.events.push_back({"TraceMe:op2", 100300, 100});
  track.events.push_back({"TraceMe:op3", 100500, 300});

  TraceProcessor processor(&trace, /*group_tiny_events=*/true);
  processor.Process();

  ASSERT_EQ(track.events.size(), 1);
  EXPECT_EQ(track.events[0].name, "3 events are hidden");
  EXPECT_EQ(track.events[0].timestamp_ps, 100000);
  EXPECT_EQ(track.events[0].duration_ps, 800);
  EXPECT_EQ(track.name, "4. XLA TraceMe");
}

}  // namespace
}  // namespace xprof::megascale
