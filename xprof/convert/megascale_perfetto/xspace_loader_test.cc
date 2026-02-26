#include "xprof/convert/megascale_perfetto/xspace_loader.h"

#include <vector>

#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {
namespace {

using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;

TEST(XSpaceLoaderTest, LoadBasic) {
  XSpace xspace;

  // Add two TPU planes with Steps, XLA Modules, and XLA Ops.
  auto build_tpu_plane = [&](XPlaneBuilder& tpu) {
    XLineBuilder steps = tpu.GetOrCreateLine(0);
    steps.SetName("Steps");
    XEventBuilder step_event =
        steps.AddEvent(*tpu.GetOrCreateEventMetadata("step 1"));
    step_event.SetTimestampPs(100);
    step_event.SetDurationPs(1000000000000LL);  // 1s

    XLineBuilder modules = tpu.GetOrCreateLine(1);
    modules.SetName("XLA Modules");
    XEventBuilder module_event =
        modules.AddEvent(*tpu.GetOrCreateEventMetadata("module 1"));
    module_event.SetTimestampPs(100);
    module_event.SetDurationPs(1000000000000LL);  // 1s

    XLineBuilder ops = tpu.GetOrCreateLine(2);
    ops.SetName("XLA Ops");
    XEventBuilder op_event1 =
        ops.AddEvent(*tpu.GetOrCreateEventMetadata("send.123"));
    op_event1.SetTimestampPs(100);
    op_event1.SetDurationPs(400000000000LL);  // 400ms

    XEventBuilder op_event2 =
        ops.AddEvent(*tpu.GetOrCreateEventMetadata("send.124"));
    op_event2.SetTimestampPs(400000000200LL);
    op_event2.SetDurationPs(500000000000LL);  // 500ms
  };

  XPlaneBuilder tpu0(xspace.add_planes());
  tpu0.SetName("/device:TPU:0");
  build_tpu_plane(tpu0);

  XPlaneBuilder tpu1(xspace.add_planes());
  tpu1.SetName("/device:TPU:1");
  build_tpu_plane(tpu1);

  // Add a Megascale plane
  XPlaneBuilder megascale(xspace.add_planes());
  megascale.SetName("/device:CUSTOM:Megascale Trace");
  XLineBuilder ml = megascale.GetOrCreateLine(0);

  // Event for TPU 0 (assume raw device ID 200000 maps to TPU 0)
  XEventBuilder me0 =
      ml.AddEvent(*megascale.GetOrCreateEventMetadata("NetworkSend END"));
  me0.SetTimestampPs(110);
  me0.SetDurationPs(20);
  me0.AddStatValue(*megascale.GetOrCreateStatMetadata("graph_key"),
                   "device_200000_gid_rendezvous1$1^i0");

  // Event for TPU 1 (assume raw device ID 200001 maps to TPU 1)
  XEventBuilder me1 =
      ml.AddEvent(*megascale.GetOrCreateEventMetadata("NetworkReceive END"));
  me1.SetTimestampPs(210);
  me1.SetDurationPs(20);
  me1.AddStatValue(*megascale.GetOrCreateStatMetadata("graph_key"),
                   "device_200001_gid_rendezvous2$1^i0");

  // Add some other CPU plane that should be filtered out
  XPlaneBuilder cpu(xspace.add_planes());
  cpu.SetName("/device:CPU:0");
  XLineBuilder cl = cpu.GetOrCreateLine(0);
  XEventBuilder cpu_event =
      cl.AddEvent(*cpu.GetOrCreateEventMetadata("cpu event"));
  cpu_event.SetTimestampPs(100);
  cpu_event.SetDurationPs(100000000000LL);  // 100ms
  cpu_event.AddStatValue(*cpu.GetOrCreateStatMetadata("stat_key"),
                         "stat_value");

  XprofTrace trace = XSpaceLoader::Load(xspace);

  // Verify TPU fragments
  EXPECT_EQ(trace.tpu_fragments.size(), 2);
  ASSERT_TRUE(trace.tpu_fragments.contains(0));
  ASSERT_TRUE(trace.tpu_fragments.contains(1));

  for (int tpu_id = 0; tpu_id < 2; ++tpu_id) {
    const auto& tracks = trace.tpu_fragments.at(tpu_id);
    EXPECT_EQ(tracks.size(), 3);
    EXPECT_EQ(tracks[0].name, "Steps");
    EXPECT_EQ(tracks[1].name, "XLA Modules");
    EXPECT_EQ(tracks[2].name, "XLA Ops");

    EXPECT_EQ(tracks[0].events.size(), 1);
    EXPECT_EQ(tracks[0].events[0].duration_ps, 1000000000000LL);

    EXPECT_EQ(tracks[1].events.size(), 1);
    EXPECT_EQ(tracks[1].events[0].duration_ps, 1000000000000LL);

    EXPECT_EQ(tracks[2].events.size(), 2);
    EXPECT_EQ(tracks[2].events[0].duration_ps, 400000000000LL);
    EXPECT_EQ(tracks[2].events[1].duration_ps, 500000000000LL);

    EXPECT_EQ(tracks[2].events[0].name, "send.123");
    EXPECT_EQ(tracks[2].events[1].name, "send.124");
  }

  // Verify Megascale fragments
  EXPECT_EQ(trace.megascale_fragments.size(), 2);
  ASSERT_TRUE(trace.megascale_fragments.contains(0));
  ASSERT_TRUE(trace.megascale_fragments.contains(1));

  EXPECT_EQ(trace.megascale_fragments.at(0).size(), 1);
  EXPECT_EQ(trace.megascale_fragments.at(0)[0].name,
            "device_200000_gid_rendezvous1");
  EXPECT_EQ(trace.megascale_fragments.at(0)[0].events.size(), 1);
  EXPECT_EQ(trace.megascale_fragments.at(0)[0].events[0].name,
            "NetworkSend END");

  EXPECT_EQ(trace.megascale_fragments.at(1).size(), 1);
  EXPECT_EQ(trace.megascale_fragments.at(1)[0].name,
            "device_200001_gid_rendezvous2");
  EXPECT_EQ(trace.megascale_fragments.at(1)[0].events.size(), 1);
  EXPECT_EQ(trace.megascale_fragments.at(1)[0].events[0].name,
            "NetworkReceive END");
}

}  // namespace
}  // namespace xprof::megascale
