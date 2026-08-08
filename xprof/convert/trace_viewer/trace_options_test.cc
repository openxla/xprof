#include "xprof/convert/trace_viewer/trace_options.h"

#include <cstdint>
#include <memory>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_set.h"
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
  EXPECT_FALSE(options.enable_legacy_dcn);

  tool_options["full_dma"] = true;
  tool_options["enable_legacy_dcn"] = true;
  options = TraceOptionsFromToolOptions(tool_options);
  EXPECT_TRUE(options.full_dma);
  EXPECT_TRUE(options.enable_legacy_dcn);

  tool_options["full_dma"] = false;
  tool_options["enable_legacy_dcn"] = false;
  options = TraceOptionsFromToolOptions(tool_options);
  EXPECT_FALSE(options.full_dma);
  EXPECT_FALSE(options.enable_legacy_dcn);
}

TEST(TraceOptionsTest, TraceOptionsToDetailsTest) {
  TraceOptions options;
  options.full_dma = true;

  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kUnknownDevice, options),
              IsEmpty());

  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kTpu, options),
              UnorderedElementsAre(Pair("full_dma", true)));

  EXPECT_THAT(TraceOptionsToDetails(TraceDeviceType::kGpu, options), IsEmpty());

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
      CreateTraceEventsFilterFromTraceOptions(options);
  EXPECT_NE(filter, nullptr);
}

TEST(TraceEventsFilterTest, FilterTest) {
  TraceOptions options;
  options.full_dma = false;
  std::unique_ptr<TraceEventsFilterInterface> filter =
      CreateTraceEventsFilterFromTraceOptions(options);

  Trace trace;
  Device& tpu_device = (*trace.mutable_devices())[0];
  tpu_device.set_device_id(0);
  tpu_device.set_name("/device:TPU:0");
  Device& tpu_noncore_device = (*trace.mutable_devices())[1];
  tpu_noncore_device.set_device_id(1);
  tpu_noncore_device.set_name("/device:TPU_COMPILER:0");

  filter->SetUp(trace);

  // A flow event on a TPU device should be filtered if full_dma is false.
  TraceEvent flow_event;
  flow_event.set_device_id(0);
  flow_event.set_flow_id(123);
  flow_event.set_flow_entry_type(TraceEvent::FLOW_MID);
  EXPECT_TRUE(filter->Filter(flow_event));

  // A non-flow event should not be filtered.
  TraceEvent non_flow_event;
  non_flow_event.set_device_id(0);
  EXPECT_FALSE(filter->Filter(non_flow_event));

  // With full_dma=true, no events should be filtered.
  options.full_dma = true;
  std::unique_ptr<TraceEventsFilterInterface> full_dma_filter =
      CreateTraceEventsFilterFromTraceOptions(options);
  full_dma_filter->SetUp(trace);
  EXPECT_FALSE(full_dma_filter->Filter(flow_event));
  EXPECT_FALSE(full_dma_filter->Filter(non_flow_event));
}

TEST(TraceEventsFilterTest, NonTpuTraceTest) {
  TraceOptions options;
  std::unique_ptr<TraceEventsFilterInterface> filter =
      CreateTraceEventsFilterFromTraceOptions(options);

  Trace trace;
  Device& gpu_device = (*trace.mutable_devices())[0];
  gpu_device.set_device_id(0);
  gpu_device.set_name("/device:GPU:0");

  filter->SetUp(trace);

  // No events should be filtered for non-TPU traces.
  TraceEvent flow_event;
  flow_event.set_device_id(0);
  flow_event.set_flow_id(123);
  flow_event.set_flow_entry_type(TraceEvent::FLOW_MID);
  EXPECT_FALSE(filter->Filter(flow_event));

  TraceEvent non_flow_event;
  non_flow_event.set_device_id(0);
  EXPECT_FALSE(filter->Filter(non_flow_event));
}

TEST(TraceEventsFilterTest, ProcessAndThreadFilterTest) {
  TraceOptions options;
  tensorflow::profiler::TraceFilterConfig config;
  config.add_device_regexes("GPU");
  config.add_resource_regexes("Thread1");
  options.trace_filter_config = config;

  std::unique_ptr<TraceEventsFilterInterface> filter =
      CreateTraceEventsFilterFromTraceOptions(options);

  Trace trace;
  Device& device1 = (*trace.mutable_devices())[0];
  device1.set_device_id(0);
  device1.set_name("/device:GPU:0");
  Resource& res1 = (*device1.mutable_resources())[10];
  res1.set_resource_id(10);
  res1.set_name("Thread1");
  Resource& res2 = (*device1.mutable_resources())[20];
  res2.set_resource_id(20);
  res2.set_name("Thread2");

  Device& device2 = (*trace.mutable_devices())[1];
  device2.set_device_id(1);
  device2.set_name("/device:CPU:0");
  Resource& res3 = (*device2.mutable_resources())[30];
  res3.set_resource_id(30);
  res3.set_name("Thread1");

  filter->SetUp(trace);

  // Match device & Match resource -> EXPECT_FALSE (not filtered)
  TraceEvent ev1;
  ev1.set_device_id(0);
  ev1.set_resource_id(10);
  EXPECT_FALSE(filter->Filter(ev1));

  // Match device & Mismatch resource -> EXPECT_TRUE (filtered)
  TraceEvent ev2;
  ev2.set_device_id(0);
  ev2.set_resource_id(20);
  EXPECT_TRUE(filter->Filter(ev2));

  // Mismatch device & Match resource -> EXPECT_TRUE (filtered)
  TraceEvent ev3;
  ev3.set_device_id(1);
  ev3.set_resource_id(30);
  EXPECT_TRUE(filter->Filter(ev3));
}

TEST(TraceEventsFilterTest, EventFilterTest) {
  TraceOptions options;
  tensorflow::profiler::TraceFilterConfig config;

  // Event Name regex filter
  tensorflow::profiler::TraceEventFilter* filter_name =
      config.add_trace_event_filters();
  filter_name->set_field_name("name");
  filter_name->set_op_id(tensorflow::profiler::TraceEventFilter::OP_REGEX);
  filter_name->set_regex_value("matmul");

  // Event Duration filter (duration >= 5.0 ms)
  tensorflow::profiler::TraceEventFilter* filter_duration =
      config.add_trace_event_filters();
  filter_duration->set_field_name("duration");
  filter_duration->set_op_id(tensorflow::profiler::TraceEventFilter::OP_GE);
  filter_duration->set_double_value(5.0);

  options.trace_filter_config = config;

  std::unique_ptr<TraceEventsFilterInterface> filter =
      CreateTraceEventsFilterFromTraceOptions(options);

  Trace trace;
  // Set up name table for testing name_ref lookup
  trace.mutable_name_table()->insert({1, "matmul_op"});
  trace.mutable_name_table()->insert({2, "convolution_op"});

  filter->SetUp(trace);

  // Match name & Match duration -> EXPECT_FALSE (kept)
  TraceEvent ev1;
  ev1.set_name_ref(1);                             // matmul_op
  ev1.set_duration_ps(6ULL * 1000 * 1000 * 1000);  //  6ms
  EXPECT_FALSE(filter->Filter(ev1));

  // Match name & Mismatch duration -> EXPECT_TRUE (filtered)
  TraceEvent ev2;
  ev2.set_name_ref(1);                             // matmul_op
  ev2.set_duration_ps(4ULL * 1000 * 1000 * 1000);  //  4ms
  EXPECT_TRUE(filter->Filter(ev2));

  // Mismatch name & Match duration -> EXPECT_TRUE (filtered)
  TraceEvent ev3;
  ev3.set_name_ref(2);                              // convolution_op
  ev3.set_duration_ps(10ULL * 1000 * 1000 * 1000);  //  10ms
  EXPECT_TRUE(filter->Filter(ev3));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
