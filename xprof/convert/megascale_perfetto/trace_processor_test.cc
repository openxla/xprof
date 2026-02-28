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
      {"device_0_gid_rendezvous$1^i0", /*ts=*/100, /*duration=*/1000});
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
  event.args.push_back({trace.string_table.Intern("buffer_sizes"),
                        trace.string_table.Intern("$c0=1000")});
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

TEST(TraceProcessorTest, ModifiesTrackNames) {
  XprofTrace trace;
  trace.tpu_fragments[0].emplace_back(Track{"Steps", {}});
  trace.megascale_fragments[0].emplace_back(
      Track{"device_0_gid_rendezvous", {}});

  TraceProcessor processor(&trace);
  processor.Process();

  EXPECT_EQ(trace.tpu_fragments[0][0].name, "1. Steps");
  EXPECT_EQ(trace.megascale_fragments[0][0].name, "rendezvous (0)");
}

}  // namespace
}  // namespace xprof::megascale
