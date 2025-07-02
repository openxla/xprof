#include "xprof/convert/trace_viewer/trace_options.h"
#include <memory>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(TraceOptionsTest, TraceOptionsFromToolOptionsTest) {
  ToolOptions tool_options;
  TraceOptions options = TraceOptionsFromToolOptions(tool_options);
  EXPECT_FALSE(options.full_dma);

  tool_options["full_dma"] = true;
  options = TraceOptionsFromToolOptions(tool_options);
  EXPECT_TRUE(options.full_dma);

  tool_options["full_dma"] = false;
  options = TraceOptionsFromToolOptions(tool_options);
  EXPECT_FALSE(options.full_dma);
}

TEST(TraceOptionsTest, TraceOptionsToDetailsTest) {
  TraceOptions options;
  options.full_dma = true;

  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kUnknownDevice, options),
              IsEmpty());

  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kTpu, options),
              UnorderedElementsAre(Pair("full_dma", true)));

  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kGpu, options),
              IsEmpty());

  options.full_dma = false;
  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kTpu, options),
              UnorderedElementsAre(Pair("full_dma", false)));
}

TEST(TraceOptionsTest, IsTpuTraceTest) {
  Trace trace;
  EXPECT_FALSE(IsTpuTrace(trace));

  Device& device = (*trace.mutable_devices())[0];
  device.set_device_id(0);
  device.set_name("/device:TPU:0");
  EXPECT_TRUE(IsTpuTrace(trace));

  device.set_name("/device:GPU:0");
  EXPECT_FALSE(IsTpuTrace(trace));
}

TEST(TraceOptionsTest, TraceEventsFilterFromTraceOptionsTest) {
  TraceOptions options;
  std::unique_ptr<TraceEventsFilterInterface> filter =
      TraceEventsFilterFromTraceOptions(options);
  EXPECT_NE(filter, nullptr);
}

TEST(TraceEventsFilterTest, FilterTest) {
  TraceOptions options;
  options.full_dma = false;
  TraceEventsFilter filter(options);

  Trace trace;
  Device& tpu_device = (*trace.mutable_devices())[0];
  tpu_device.set_device_id(0);
  tpu_device.set_name("/device:TPU:0");
  Device& tpu_noncore_device = (*trace.mutable_devices())[1];
  tpu_noncore_device.set_device_id(1);
  tpu_noncore_device.set_name("/device:TPU_COMPILER:0");

  filter.SetUp(trace);

  // A flow event on a TPU device should be filtered if full_dma is false.
  TraceEvent flow_event;
  flow_event.set_device_id(0);
  flow_event.set_flow_id(123);
  flow_event.set_flow_entry_type(TraceEvent::FLOW_MID);
  EXPECT_TRUE(filter.Filter(flow_event));

  // A non-flow event should not be filtered.
  TraceEvent non_flow_event;
  non_flow_event.set_device_id(0);
  EXPECT_FALSE(filter.Filter(non_flow_event));

  // With full_dma=true, no events should be filtered.
  options.full_dma = true;
  TraceEventsFilter full_dma_filter(options);
  full_dma_filter.SetUp(trace);
  EXPECT_FALSE(full_dma_filter.Filter(flow_event));
  EXPECT_FALSE(full_dma_filter.Filter(non_flow_event));
}

TEST(TraceEventsFilterTest, NonTpuTraceTest) {
  TraceOptions options;
  TraceEventsFilter filter(options);

  Trace trace;
  Device& gpu_device = (*trace.mutable_devices())[0];
  gpu_device.set_device_id(0);
  gpu_device.set_name("/device:GPU:0");

  filter.SetUp(trace);

  // No events should be filtered for non-TPU traces.
  TraceEvent flow_event;
  flow_event.set_device_id(0);
  flow_event.set_flow_id(123);
  flow_event.set_flow_entry_type(TraceEvent::FLOW_MID);
  EXPECT_FALSE(filter.Filter(flow_event));

  TraceEvent non_flow_event;
  non_flow_event.set_device_id(0);
  EXPECT_FALSE(filter.Filter(non_flow_event));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
