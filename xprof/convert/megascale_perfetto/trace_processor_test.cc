#include "xprof/convert/megascale_perfetto/trace_processor.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "<gtest/gtest.h>"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {
namespace {

TEST(TraceProcessorTest, SortsEvents) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "XLA Ops";

  // Unsorted events
  track.events.push_back({"op 1", /*ts=*/100, /*duration=*/50});
  track.events.push_back({"op 2", /*ts=*/200, /*duration=*/50});
  // If timestamps are the same, longer duration should come first.
  track.events.push_back({"op 3", /*ts=*/200, /*duration=*/100});
  track.events.push_back({"op 4", /*ts=*/50, /*duration=*/100});

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

}  // namespace
}  // namespace xprof::megascale
