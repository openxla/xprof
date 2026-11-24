#include "xprof/convert/megascale_perfetto/perfetto_writer.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/cord.h"
#include "protos/perfetto/trace/trace.pb.h"
#include "protos/perfetto/trace/trace_packet.pb.h"
#include "protos/perfetto/trace/track_event/track_descriptor.pb.h"
#include "protos/perfetto/trace/track_event/track_event.pb.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {
namespace {

TEST(PerfettoWriterTest, WriteToCordBasic) {
  XprofTrace trace;

  // Minimal data
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "Steps";
  track.events.push_back({"step 1", 1000000, 500000});  // 1us, 0.5us

  absl::Cord output;
  ASSERT_OK(
      PerfettoWriter::WriteToCord(trace, &output, /*compressed_output=*/false));

  perfetto::protos::Trace trace_proto;
  ASSERT_TRUE(trace_proto.ParseFromString(output));

  // Verify basic structure
  bool found_track_descriptor = false;
  bool found_track_event = false;

  for (const auto& packet : trace_proto.packet()) {
    if (packet.has_track_descriptor()) {
      if (packet.track_descriptor().name() == "Steps") {
        found_track_descriptor = true;
      }
    }
    if (packet.has_track_event()) {
      if (packet.track_event().type() ==
          perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN) {
        found_track_event = true;
      }
    }
  }

  EXPECT_TRUE(found_track_descriptor);
  EXPECT_TRUE(found_track_event);
}

TEST(PerfettoWriterTest, WriteToStringBasic) {
  XprofTrace trace;
  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "TestTrack";
  track.events.push_back({"event1", 1000, 500});

  std::string output;
  ASSERT_OK(PerfettoWriter::WriteToString(trace, &output,
                                          /*compressed_output=*/false));

  perfetto::protos::Trace trace_proto;
  ASSERT_TRUE(trace_proto.ParseFromString(output));

  bool found = false;
  for (const auto& packet : trace_proto.packet()) {
    if (packet.has_track_descriptor() &&
        packet.track_descriptor().name() == "TestTrack") {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(PerfettoWriterTest, ClampsNegativeTimestampsToZero) {
  XprofTrace trace;
  trace.rx_counter.name = "Outstanding Bytes RX";
  trace.rx_counter.timestamps_ps = {-1400000000, 600000000};
  trace.rx_counter.values = {16777216, 0};

  Track& track = trace.tpu_fragments[0].emplace_back();
  track.name = "Steps";
  track.events.push_back({"early_slice", -500000, 1000000});

  std::string output;
  ASSERT_OK(PerfettoWriter::WriteToString(trace, &output,
                                          /*compressed_output=*/false));

  perfetto::protos::Trace trace_proto;
  ASSERT_TRUE(trace_proto.ParseFromString(output));

  std::vector<uint64_t> counter_timestamps;
  std::vector<uint64_t> slice_timestamps;
  for (const auto& packet : trace_proto.packet()) {
    if (!packet.has_track_event()) {
      continue;
    }
    if (packet.track_event().type() ==
        perfetto::protos::TrackEvent::TYPE_COUNTER) {
      counter_timestamps.push_back(packet.timestamp());
    } else if (packet.track_event().type() ==
                   perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN ||
               packet.track_event().type() ==
                   perfetto::protos::TrackEvent::TYPE_SLICE_END) {
      slice_timestamps.push_back(packet.timestamp());
    }
  }

  ASSERT_EQ(counter_timestamps.size(), 2);
  EXPECT_EQ(counter_timestamps[0], 0);
  EXPECT_EQ(counter_timestamps[1], 600000);

  ASSERT_EQ(slice_timestamps.size(), 2);
  EXPECT_EQ(slice_timestamps[0], 0);
  EXPECT_EQ(slice_timestamps[1], 500);
}

}  // namespace
}  // namespace xprof::megascale
