#include "xprof/convert/megascale_perfetto/perfetto_writer.h"

#include <string>
#include <vector>

#include "<gtest/gtest.h>"
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

}  // namespace
}  // namespace xprof::megascale
