#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::EqualsProto;
using ::testing::proto::Partially;

class TestTraceEventsContainer
    : public TraceEventsContainerBase<EventFactory, RawData> {
 public:
  explicit TestTraceEventsContainer(const Trace& trace) { trace_ = trace; }

  void AddCounterEvent(uint32_t device_id, absl::string_view name,
                       TraceEvent* event) {
    event->set_device_id(device_id);
    event->set_name(name.data(), name.size());
    AddArenaEvent(event);
  }

  void AddCompleteEvent(uint32_t device_id, uint64_t resource_id,
                        TraceEvent* event) {
    event->set_device_id(device_id);
    event->set_resource_id(resource_id);
    AddArenaEvent(event);
  }

  void AddAsyncEvent(uint32_t device_id, absl::string_view name,
                     TraceEvent* event) {
    event->set_device_id(device_id);
    event->set_name(name.data(), name.size());
    AddArenaEvent(event);
  }

  const Trace& trace() const { return trace_; }
};

class StringOutput {
 public:
  void WriteString(absl::string_view source) {
    str_.append(source.data(), source.size());
  }
  const std::string& str() const { return str_; }

 private:
  std::string str_;
};

TEST(DeltaSeriesProtoConverterTest, ConvertsCompleteEventsAndDeltas) {
  Trace trace;
  // Setup Device and Resource
  Device device;
  device.set_name("CPU");
  Resource resource;
  resource.set_name("Thread 1");
  (*device.mutable_resources())[1] = resource;
  (*trace.mutable_devices())[0] = device;

  // Setup Trace events
  TraceEvent event1;
  event1.set_device_id(0);
  event1.set_resource_id(1);
  event1.set_name("Compute");
  event1.set_timestamp_ps(1000);
  event1.set_duration_ps(500);

  event1.set_serial(42);

  TraceEvent event2;
  event2.set_device_id(0);
  event2.set_resource_id(1);
  event2.set_name("Compute");
  event2.set_timestamp_ps(2000);
  event2.set_duration_ps(300);

  TestTraceEventsContainer container(trace);
  container.AddCompleteEvent(0, 1, &event1);
  container.AddCompleteEvent(0, 1, &event2);

  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));

  // Decompress to verify the structure
  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.complete_events_size(), 1);
  const auto& series = response.complete_events(0);

  EXPECT_EQ(series.metadata().process_id(), 0);
  EXPECT_EQ(series.metadata().thread_id(), 1);

  ASSERT_EQ(series.deltas_size(), 2);
  // First element is the absolute start timestamp
  EXPECT_EQ(series.deltas(0), 1000);
  // Second element is diff from previous timestamp (2000 - 1000)
  EXPECT_EQ(series.deltas(1), 1000);

  ASSERT_EQ(series.durations_size(), 2);
  EXPECT_EQ(series.durations(0), 500);
  EXPECT_EQ(series.durations(1), 300);

  ASSERT_EQ(series.event_metadata_size(), 2);
  EXPECT_EQ(series.event_metadata(0).serial(), 42);
  // Default is 0 when unset
  EXPECT_EQ(series.event_metadata(1).serial(), 0);

  EXPECT_EQ(response.metadata().processes_size(), 1);
  const auto& process = response.metadata().processes(0);
  EXPECT_EQ(process.name(), "CPU");
  ASSERT_EQ(process.threads_size(), 1);
  EXPECT_EQ(process.threads(0).name(), "Thread 1");
}

TEST(DeltaSeriesProtoConverterTest, ConvertsCounterEvents) {
  Trace trace;
  Device device;
  (*trace.mutable_devices())[0] = device;

  TraceEvent event1;
  event1.set_device_id(0);
  event1.set_name("MyCounter");
  event1.set_timestamp_ps(1000);

  // Set counter value to 1234 via RawData
  RawData raw_data1;
  raw_data1.mutable_args()->add_arg()->set_uint_value(1234);
  event1.set_raw_data(raw_data1.SerializeAsString());

  TraceEvent event2;
  event2.set_device_id(0);
  event2.set_name("MyCounter");
  event2.set_timestamp_ps(1500);

  // Set counter value to 5678 via RawData
  RawData raw_data2;
  raw_data2.mutable_args()->add_arg()->set_uint_value(5678);
  event2.set_raw_data(raw_data2.SerializeAsString());

  TestTraceEventsContainer container(trace);
  container.AddCounterEvent(0, "MyCounter", &event1);
  container.AddCounterEvent(0, "MyCounter", &event2);

  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.counter_events_size(), 1);
  const auto& series = response.counter_events(0);

  EXPECT_EQ(series.metadata().process_id(), 0);

  ASSERT_EQ(series.deltas_size(), 2);
  EXPECT_EQ(series.deltas(0), 1000);
  EXPECT_EQ(series.deltas(1), 500);

  ASSERT_EQ(series.event_metadata_size(), 2);
  EXPECT_EQ(series.event_metadata(0).counter_value_uint64(), 1234);
  EXPECT_EQ(series.event_metadata(1).counter_value_uint64(), 5678);
  EXPECT_FALSE(series.metadata().has_event_stats_ref());
}

TEST(DeltaSeriesProtoConverterTest, ConvertsCounterEventsWithSeriesName) {
  Trace trace;
  Device device;
  (*trace.mutable_devices())[0] = device;

  TraceEvent event1;
  event1.set_device_id(0);
  event1.set_name("MyCounter");
  event1.set_timestamp_ps(1000);

  RawData raw_data1;
  auto* arg1 = raw_data1.mutable_args()->add_arg();
  arg1->set_name("MiB");
  arg1->set_double_value(12.5);
  event1.set_raw_data(raw_data1.SerializeAsString());

  TraceEvent event2;
  event2.set_device_id(0);
  event2.set_name("MyCounter");
  event2.set_timestamp_ps(1500);

  RawData raw_data2;
  auto* arg2 = raw_data2.mutable_args()->add_arg();
  arg2->set_name("MiB");
  arg2->set_double_value(25.0);
  event2.set_raw_data(raw_data2.SerializeAsString());

  TestTraceEventsContainer container(trace);
  container.AddCounterEvent(0, "MyCounter", &event1);
  container.AddCounterEvent(0, "MyCounter", &event2);

  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.counter_events_size(), 1);
  const auto& series = response.counter_events(0);

  EXPECT_EQ(series.metadata().process_id(), 0);
  EXPECT_TRUE(series.metadata().has_event_stats_ref());
  ASSERT_LT(series.metadata().event_stats_ref(),
            response.interned_strings_size());
  EXPECT_EQ(response.interned_strings(series.metadata().event_stats_ref()),
            "MiB");

  ASSERT_EQ(series.deltas_size(), 2);
  EXPECT_EQ(series.deltas(0), 1000);
  EXPECT_EQ(series.deltas(1), 500);

  ASSERT_EQ(series.event_metadata_size(), 2);
  EXPECT_DOUBLE_EQ(series.event_metadata(0).counter_value_double(), 12.5);
  EXPECT_DOUBLE_EQ(series.event_metadata(1).counter_value_double(), 25.0);
}

TEST(DeltaSeriesProtoConverterTest, ConvertsAsyncEvents) {
  Trace trace;
  Device device;
  (*trace.mutable_devices())[0] = device;

  TraceEvent event1;
  event1.set_device_id(0);
  event1.set_name("AsyncOp");
  event1.set_timestamp_ps(1000);
  event1.set_duration_ps(500);
  event1.set_flow_id(1001);
  event1.set_flow_category(2);  // 2 usually corresponds to kTfExecutor
  event1.set_group_id(42);

  TraceEvent event2;
  event2.set_device_id(0);
  event2.set_name("AsyncOp");
  event2.set_timestamp_ps(2000);
  event2.set_duration_ps(100);
  event2.set_flow_id(1002);
  event2.set_flow_category(2);
  event2.set_group_id(42);

  TestTraceEventsContainer container(trace);
  container.AddAsyncEvent(0, "AsyncOp", &event1);
  container.AddAsyncEvent(0, "AsyncOp", &event2);

  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.async_events_size(), 1);
  const auto& series = response.async_events(0);

  EXPECT_EQ(series.metadata().process_id(), 0);

  ASSERT_EQ(series.deltas_size(), 2);
  EXPECT_EQ(series.deltas(0), 1000);
  EXPECT_EQ(series.deltas(1), 1000);

  ASSERT_EQ(series.durations_size(), 2);
  EXPECT_EQ(series.durations(0), 500);
  EXPECT_EQ(series.durations(1), 100);

  ASSERT_EQ(series.event_metadata_size(), 2);
  const auto& metadata = series.event_metadata(0);
  EXPECT_EQ(metadata.flow_id(), 1001);
  EXPECT_EQ(metadata.group_id(), 42);

  EXPECT_GT(metadata.flow_category(), 0);
  ASSERT_LT(metadata.flow_category(), response.interned_strings_size());

  const auto& metadata2 = series.event_metadata(1);
  EXPECT_EQ(metadata2.flow_id(), 1002);
  EXPECT_EQ(metadata2.flow_category(), metadata.flow_category());
}

TEST(DeltaSeriesProtoConverterTest, ConvertsMixedEvents) {
  Trace trace;
  Device device;
  Resource resource;
  (*device.mutable_resources())[1] = resource;
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  // 1. Complete Events
  TraceEvent complete_event1;
  complete_event1.set_device_id(0);
  complete_event1.set_resource_id(1);
  complete_event1.set_name("Compute");
  complete_event1.set_timestamp_ps(1000);
  complete_event1.set_duration_ps(100);
  container.AddCompleteEvent(0, 1, &complete_event1);

  TraceEvent complete_event2;
  complete_event2.set_device_id(0);
  complete_event2.set_resource_id(1);
  complete_event2.set_name("Compute");
  complete_event2.set_timestamp_ps(1200);
  complete_event2.set_duration_ps(200);
  container.AddCompleteEvent(0, 1, &complete_event2);

  // 2. Async Events
  TraceEvent async_event1;
  async_event1.set_device_id(0);
  async_event1.set_name("AsyncOp");
  async_event1.set_timestamp_ps(1500);
  async_event1.set_duration_ps(200);
  async_event1.set_flow_id(1);
  container.AddAsyncEvent(0, "AsyncOp", &async_event1);

  TraceEvent async_event2;
  async_event2.set_device_id(0);
  async_event2.set_name("AsyncOp");
  async_event2.set_timestamp_ps(1800);
  async_event2.set_duration_ps(100);
  async_event2.set_flow_id(2);
  container.AddAsyncEvent(0, "AsyncOp", &async_event2);

  // 3. Counter Events
  TraceEvent counter_event1;
  counter_event1.set_device_id(0);
  counter_event1.set_name("Memory");
  counter_event1.set_timestamp_ps(2000);
  RawData raw_data1;
  raw_data1.mutable_args()->add_arg()->set_double_value(3.14);
  counter_event1.set_raw_data(raw_data1.SerializeAsString());
  container.AddCounterEvent(0, "Memory", &counter_event1);

  TraceEvent counter_event2;
  counter_event2.set_device_id(0);
  counter_event2.set_name("Memory");
  counter_event2.set_timestamp_ps(2500);
  RawData raw_data2;
  raw_data2.mutable_args()->add_arg()->set_double_value(6.28);
  counter_event2.set_raw_data(raw_data2.SerializeAsString());
  container.AddCounterEvent(0, "Memory", &counter_event2);

  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  EXPECT_EQ(response.complete_events_size(), 1);
  EXPECT_EQ(response.async_events_size(), 1);
  EXPECT_EQ(response.counter_events_size(), 1);

  EXPECT_EQ(response.complete_events(0).deltas_size(), 2);
  EXPECT_EQ(response.complete_events(0).deltas(0), 1000);
  EXPECT_EQ(response.complete_events(0).deltas(1), 200);

  EXPECT_EQ(response.async_events(0).deltas_size(), 2);
  EXPECT_EQ(response.async_events(0).deltas(0), 1500);
  EXPECT_EQ(response.async_events(0).deltas(1), 300);

  EXPECT_EQ(response.counter_events(0).deltas_size(), 2);
  EXPECT_EQ(response.counter_events(0).deltas(0), 2000);
  EXPECT_EQ(response.counter_events(0).deltas(1), 500);

  EXPECT_EQ(response.counter_events(0).event_metadata(0).counter_value_double(),
            3.14);
  EXPECT_EQ(response.counter_events(0).event_metadata(1).counter_value_double(),
            6.28);
}

TEST(DeltaSeriesProtoConverterTest, HonorsMpmdPipelineView) {
  Trace trace;
  // Two devices
  Device device0;
  device0.set_name("TPU 0");
  Resource resource0;
  resource0.set_name("Thread 0");
  (*device0.mutable_resources())[1] = resource0;
  (*trace.mutable_devices())[0] = device0;

  Device device1;
  device1.set_name("TPU 1");
  Resource resource1;
  resource1.set_name("Thread 1");
  (*device1.mutable_resources())[1] = resource1;
  (*trace.mutable_devices())[1] = device1;

  // Let event on device 0 run layer 1.
  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("p2_layer_1.my_program_name(123)");

  // Let event on device 1 run layer 0.
  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(1);
  event1.set_name("p2_layer_0.my_program_name(123)");

  TestTraceEventsContainer container(trace);
  container.AddCompleteEvent(0, 1, &event0);
  container.AddCompleteEvent(1, 1, &event1);

  DeltaSeriesProtoConversionOptions options;
  options.mpmd_pipeline_view = true;
  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.metadata().processes_size(), 2);
  for (const auto& process : response.metadata().processes()) {
    if (process.id() == 0) {
      // Device 0 ran layer 1, so it should have a higher sort index than device
      // 1
      EXPECT_EQ(process.sort_index(), 1);
    } else if (process.id() == 1) {
      EXPECT_EQ(process.sort_index(), 0);
    }
  }
}

TEST(DeltaSeriesProtoConverterTest, PopulatesDetails) {
  Trace trace;
  TestTraceEventsContainer container(trace);

  DeltaSeriesProtoConversionOptions options;
  options.details.push_back({"key1", true});
  options.details.push_back({"key2", false});

  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  EXPECT_THAT(response, Partially(EqualsProto(R"pb(
                details { name: "key1" value: true }
                details { name: "key2" value: false }
              )pb")));
}

TEST(DeltaSeriesProtoConverterTest, PopulatesFullTimespan) {
  Trace trace;
  trace.set_min_timestamp_ps(12345000);
  trace.set_max_timestamp_ps(67890000);
  TestTraceEventsContainer container(trace);

  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  EXPECT_TRUE(response.has_full_timespan_start_ps());
  EXPECT_THAT(response.full_timespan_start_ps(), Eq(12345000));
  EXPECT_TRUE(response.has_full_timespan_end_ps());
  EXPECT_THAT(response.full_timespan_end_ps(), Eq(67890000));
}

TEST(DeltaSeriesProtoConverterTest, SuppressesSortIndexForCustomSortResources) {
  // Set up two devices to verify conditional sort_index suppression:
  // - Device 5 ("MPMD Custom Device") has two resources (IDs 10 and 20).
  // - Device 6 ("Standard Device") has one resource (ID 30).
  //
  // When Device 5 is added to `options.sort_resources_by_name`, its threads
  // in the generated proto must suppress explicit sort_index (enabling
  // alphabetical track ordering in the frontend), whereas Device 6 threads
  // must retain explicit numerical sort_index derived from resource IDs.
  Trace trace;
  Device device1;
  device1.set_name("MPMD Custom Device");

  Resource resource1;
  resource1.set_name("B_Program");
  (*device1.mutable_resources())[20] = resource1;

  Resource resource2;
  resource2.set_name("A_Program");
  (*device1.mutable_resources())[10] = resource2;

  (*trace.mutable_devices())[5] = device1;

  Device device2;
  device2.set_name("Standard Device");

  Resource resource3;
  resource3.set_name("Standard Resource");
  (*device2.mutable_resources())[30] = resource3;

  (*trace.mutable_devices())[6] = device2;

  TestTraceEventsContainer container(trace);

  DeltaSeriesProtoConversionOptions options;
  options.sort_resources_by_name.insert(5);  // Device ID 5 only

  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.metadata().processes_size(), 2);

  // Search processes by id() to avoid depending on map iteration order.
  bool found_device5 = false;
  bool found_device6 = false;
  for (const xprof::Process& process : response.metadata().processes()) {
    if (process.id() == 5) {
      found_device5 = true;
      EXPECT_EQ(process.name(), "MPMD Custom Device");
      ASSERT_EQ(process.threads_size(), 2);
      for (const xprof::Thread& thread : process.threads()) {
        EXPECT_FALSE(thread.has_sort_index());
      }
    } else if (process.id() == 6) {
      found_device6 = true;
      EXPECT_EQ(process.name(), "Standard Device");
      ASSERT_EQ(process.threads_size(), 1);
      EXPECT_EQ(process.threads(0).id(), 30);
      EXPECT_TRUE(process.threads(0).has_sort_index());
      EXPECT_EQ(process.threads(0).sort_index(), 30);
    }
  }

  EXPECT_TRUE(found_device5);
  EXPECT_TRUE(found_device6);
}

TEST(DeltaSeriesProtoConverterTest,
     MpmdPipelineViewOrdersDevicesWithoutStagesAfterStages) {
  Trace trace;
  // Device 0: Active TPU 0 (stage 0).
  Device device0;
  device0.set_name("host0 /device:TPU:0");
  Resource resource0;
  resource0.set_name("XLA Modules");
  (*device0.mutable_resources())[1] = resource0;
  (*trace.mutable_devices())[0] = device0;

  // Device 1: Active TPU 1 (stage 1).
  Device device1;
  device1.set_name("host0 /device:TPU:1");
  Resource resource1;
  resource1.set_name("XLA Modules");
  (*device1.mutable_resources())[1] = resource1;
  (*trace.mutable_devices())[1] = device1;

  // Device 2: TPU 2 without MPMD stage events.
  Device device2;
  device2.set_name("host0 /device:TPU:2");
  Resource resource2;
  resource2.set_name("XLA Modules");
  (*device2.mutable_resources())[1] = resource2;
  (*trace.mutable_devices())[2] = device2;

  // Device 10: Host CPU.
  Device device10;
  device10.set_name("/host:CPU:0");
  Resource resource10;
  resource10.set_name("Host Thread");
  (*device10.mutable_resources())[1] = resource10;
  (*trace.mutable_devices())[10] = device10;

  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("p0_stage0.program(1)");
  event0.set_timestamp_ps(1000);
  event0.set_duration_ps(500);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(1);
  event1.set_name("p1_stage1.program(1)");
  event1.set_timestamp_ps(2000);
  event1.set_duration_ps(500);

  TraceEvent event2;
  event2.set_device_id(2);
  event2.set_resource_id(1);
  event2.set_name("non_mpmd_compute");
  event2.set_timestamp_ps(2500);
  event2.set_duration_ps(500);

  TestTraceEventsContainer container(trace);
  container.AddCompleteEvent(0, 1, &event0);
  container.AddCompleteEvent(1, 1, &event1);
  container.AddCompleteEvent(2, 1, &event2);

  DeltaSeriesProtoConversionOptions options;
  options.mpmd_pipeline_view = true;

  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  // All 4 processes are preserved (0, 1, 2, 10).
  ASSERT_EQ(response.metadata().processes_size(), 4);

  absl::flat_hash_map<uint32_t, uint32_t> process_sort_indices;
  for (const xprof::Process& process : response.metadata().processes()) {
    process_sort_indices[process.id()] = process.sort_index();
  }

  EXPECT_TRUE(process_sort_indices.contains(0));
  EXPECT_TRUE(process_sort_indices.contains(1));
  EXPECT_TRUE(process_sort_indices.contains(2));
  EXPECT_TRUE(process_sort_indices.contains(10));

  EXPECT_EQ(process_sort_indices.at(0), 0);
  EXPECT_EQ(process_sort_indices.at(1), 1);
  EXPECT_EQ(process_sort_indices.at(2), kMpmdUnrankedSortIndexBase + 2);
  EXPECT_EQ(process_sort_indices.at(10), kMpmdUnrankedSortIndexBase + 10);

  // Verify that events on device 2 are emitted.
  bool found_device2_event = false;
  for (const xprof::TraceEventSeries& series : response.complete_events()) {
    if (series.metadata().process_id() == 2) {
      found_device2_event = true;
    }
  }
  EXPECT_TRUE(found_device2_event);
}

TEST(DeltaSeriesProtoConverterTest,
     MpmdZeroRankedDevicesOmitsProcessSortIndex) {
  Trace trace;
  (*trace.mutable_devices())[0].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[0].set_name("host0 /device:TPU:0");
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("host0 /device:TPU:1");

  TraceEvent event0;
  event0.set_timestamp_ps(100);
  event0.set_duration_ps(50);
  event0.set_name("regular_kernel");

  TestTraceEventsContainer container(trace);
  container.AddCompleteEvent(0, 1, &event0);

  DeltaSeriesProtoConversionOptions options;
  options.mpmd_pipeline_view = true;

  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.metadata().processes_size(), 2);
  for (const xprof::Process& process : response.metadata().processes()) {
    EXPECT_FALSE(process.has_sort_index());
  }
}

TEST(DeltaSeriesProtoConverterTest,
     MpmdProcessMetadataIndependentOfLoadedWindow) {
  Trace trace;
  // Device 0: TPU Core 0 (Stage 0).
  (*trace.mutable_devices())[0].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[0].set_name("host0 /device:TPU:0");

  // Device 1: TPU Core 1 (Stage 1).
  (*trace.mutable_devices())[1].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[1].set_name("host0 /device:TPU:1");

  // Device 3: Idle TPU Core 3 without module events.
  (*trace.mutable_devices())[3].mutable_resources()->operator[](1).set_name(
      "XLA Modules");
  (*trace.mutable_devices())[3].set_name("host0 /device:TPU:3");

  TraceEvent event0;
  event0.set_timestamp_ps(100);
  event0.set_duration_ps(100);
  event0.set_name("p0_stage0.program(1)");

  TraceEvent event1;
  event1.set_timestamp_ps(200);
  event1.set_duration_ps(100);
  event1.set_name("p0_stage1.program(1)");

  // Full trace (K = 2): both Stage 0 and Stage 1 active.
  TestTraceEventsContainer full_container(trace);
  full_container.AddCompleteEvent(0, 1, &event0);
  full_container.AddCompleteEvent(1, 1, &event1);

  // Windowed trace (K = 1): only Stage 0 active.
  TestTraceEventsContainer windowed_container(trace);
  windowed_container.AddCompleteEvent(0, 1, &event0);

  DeltaSeriesProtoConversionOptions options;
  options.mpmd_pipeline_view = true;

  ASSERT_OK_AND_ASSIGN(
      std::string full_compressed,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, full_container));
  ASSERT_OK_AND_ASSIGN(std::string full_decompressed,
                       ZstdCompression::Decompress(full_compressed));
  xprof::TraceDataResponse full_response;
  ASSERT_TRUE(full_response.ParseFromString(full_decompressed));

  ASSERT_OK_AND_ASSIGN(std::string windowed_compressed,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           options, windowed_container));
  ASSERT_OK_AND_ASSIGN(std::string windowed_decompressed,
                       ZstdCompression::Decompress(windowed_compressed));
  xprof::TraceDataResponse windowed_response;
  ASSERT_TRUE(windowed_response.ParseFromString(windowed_decompressed));

  absl::flat_hash_map<uint32_t, uint32_t> full_sort_indices;
  for (const xprof::Process& process : full_response.metadata().processes()) {
    full_sort_indices[process.id()] = process.sort_index();
  }

  absl::flat_hash_map<uint32_t, uint32_t> windowed_sort_indices;
  for (const xprof::Process& process :
       windowed_response.metadata().processes()) {
    windowed_sort_indices[process.id()] = process.sort_index();
  }

  // In full load (K = 2), device 0 gets 0, device 1 gets 1.
  EXPECT_EQ(full_sort_indices.at(0), 0);
  EXPECT_EQ(full_sort_indices.at(1), 1);

  // In windowed load (K = 1), device 0 gets 0, device 1 gets Base + 1.
  EXPECT_EQ(windowed_sort_indices.at(0), 0);
  EXPECT_EQ(windowed_sort_indices.at(1), kMpmdUnrankedSortIndexBase + 1);

  // For idle device 3, both full and windowed loads emit the identical
  // kMpmdUnrankedSortIndexBase + 3 sort index.
  EXPECT_EQ(full_sort_indices.at(3), kMpmdUnrankedSortIndexBase + 3);
  EXPECT_EQ(windowed_sort_indices.at(3), kMpmdUnrankedSortIndexBase + 3);
}

TEST(DeltaSeriesProtoConverterTest,
     MpmdSingleDevicePerStageDeduplicatesStages) {
  Trace trace;
  // Device 0: TPU 0 running stage 0 (representative).
  Device device0;
  device0.set_name("host0 /device:TPU:0");
  Resource resource0;
  resource0.set_name("XLA Modules");
  (*device0.mutable_resources())[1] = resource0;
  (*trace.mutable_devices())[0] = device0;

  // Device 1: TPU 1 running stage 0 (duplicate, should be pruned).
  Device device1;
  device1.set_name("host0 /device:TPU:1");
  Resource resource1;
  resource1.set_name("XLA Modules");
  (*device1.mutable_resources())[1] = resource1;
  (*trace.mutable_devices())[1] = device1;

  // Device 2: TPU 2 running stage 1 (representative).
  Device device2;
  device2.set_name("host0 /device:TPU:2");
  Resource resource2;
  resource2.set_name("XLA Modules");
  (*device2.mutable_resources())[1] = resource2;
  (*trace.mutable_devices())[2] = device2;

  // Device 10: Host CPU.
  Device device10;
  device10.set_name("/host:CPU:0");
  Resource resource10;
  resource10.set_name("Host Thread");
  (*device10.mutable_resources())[1] = resource10;
  (*trace.mutable_devices())[10] = device10;

  TraceEvent event0;
  event0.set_device_id(0);
  event0.set_resource_id(1);
  event0.set_name("p0_stage0.program(1)");
  event0.set_timestamp_ps(1000);
  event0.set_duration_ps(500);

  TraceEvent event1;
  event1.set_device_id(1);
  event1.set_resource_id(1);
  event1.set_name("p0_stage0.program(1)");
  event1.set_timestamp_ps(1000);
  event1.set_duration_ps(500);

  TraceEvent event2;
  event2.set_device_id(2);
  event2.set_resource_id(1);
  event2.set_name("p1_stage1.program(1)");
  event2.set_timestamp_ps(2000);
  event2.set_duration_ps(500);

  TestTraceEventsContainer container(trace);
  container.AddCompleteEvent(0, 1, &event0);
  container.AddCompleteEvent(1, 1, &event1);
  container.AddCompleteEvent(2, 1, &event2);

  DeltaSeriesProtoConversionOptions options;
  options.mpmd_pipeline_view = true;
  options.mpmd_single_device_per_stage = true;

  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));

  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  // Duplicate Device 1 must be pruned. Total processes: 3 (0, 2, 10).
  ASSERT_EQ(response.metadata().processes_size(), 3);

  absl::flat_hash_map<uint32_t, uint32_t> process_sort_indices;
  for (const xprof::Process& process : response.metadata().processes()) {
    process_sort_indices[process.id()] = process.sort_index();
  }

  EXPECT_TRUE(process_sort_indices.contains(0));
  EXPECT_FALSE(process_sort_indices.contains(1));
  EXPECT_TRUE(process_sort_indices.contains(2));
  EXPECT_TRUE(process_sort_indices.contains(10));

  EXPECT_EQ(process_sort_indices.at(0), 0);
  EXPECT_EQ(process_sort_indices.at(2), 1);
  // Device 10 gets fallback sort_index: kMpmdUnrankedSortIndexBase + 10.
  EXPECT_EQ(process_sort_indices.at(10), kMpmdUnrankedSortIndexBase + 10);

  // Verify that events on duplicate device 1 were not emitted.
  for (const xprof::TraceEventSeries& series : response.complete_events()) {
    EXPECT_NE(series.metadata().process_id(), 1);
  }
}

TEST(DeltaSeriesProtoConverterTest, SparseMetadataAndRleGroupIdEncoding) {
  Trace trace;
  Device device;
  Resource resource;
  (*device.mutable_resources())[1] = resource;
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  // Complete events: 3 events in group 10, 2 events in group 11; only index 2
  // has flow_id != 0 and index 4 has serial != 0.
  std::vector<TraceEvent> complete_events(5);
  for (int i = 0; i < 5; ++i) {
    TraceEvent& ev = complete_events[i];
    ev.set_device_id(0);
    ev.set_resource_id(1);
    ev.set_name("Op");
    ev.set_timestamp_ps((i + 1) * 1000);
    ev.set_duration_ps(500);
    ev.set_group_id(i < 3 ? 10 : 11);
    if (i == 2) ev.set_flow_id(777);
    if (i == 4) ev.set_serial(99);
    container.AddCompleteEvent(0, 1, &ev);
  }

  // Pre-paired async event (duration_ps > 0, serial == 0, generic category):
  // must omit TraceEventMetadata while still recording group_id RLE.
  TraceEvent pre_paired_async;
  pre_paired_async.set_device_id(0);
  pre_paired_async.set_name("AsyncXlaOp");
  pre_paired_async.set_timestamp_ps(1000);
  pre_paired_async.set_duration_ps(400);
  pre_paired_async.set_flow_id(555);
  pre_paired_async.set_group_id(10);
  container.AddAsyncEvent(0, "AsyncXlaOp", &pre_paired_async);

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.complete_events_size(), 1);
  const xprof::TraceEventSeries& comp = response.complete_events(0);
  ASSERT_EQ(comp.event_metadata_size(), 2);
  ASSERT_EQ(comp.event_metadata_indices_size(), 2);
  EXPECT_EQ(comp.event_metadata_indices(0), 2u);
  EXPECT_EQ(comp.event_metadata(0).flow_id(), 777u);
  EXPECT_EQ(comp.event_metadata_indices(1), 4u);
  EXPECT_EQ(comp.event_metadata(1).serial(), 99u);

  ASSERT_EQ(comp.group_id_run_values_size(), 2);
  ASSERT_EQ(comp.group_id_run_lengths_size(), 2);
  EXPECT_EQ(comp.group_id_run_values(0), 10);
  EXPECT_EQ(comp.group_id_run_lengths(0), 3u);
  EXPECT_EQ(comp.group_id_run_values(1), 11);
  EXPECT_EQ(comp.group_id_run_lengths(1), 2u);

  ASSERT_EQ(response.async_events_size(), 1);
  const xprof::TraceEventSeries& async_series = response.async_events(0);
  EXPECT_EQ(async_series.event_metadata_size(), 0);
  EXPECT_EQ(async_series.event_metadata_indices_size(), 0);
  ASSERT_EQ(async_series.group_id_run_values_size(), 1);
  EXPECT_EQ(async_series.group_id_run_values(0), 10);
  EXPECT_EQ(async_series.group_id_run_lengths(0), 1u);
}

TEST(DeltaSeriesProtoConverterTest,
     DefaultOptionsEmitDenseMetadataWithoutIndicesOrRle) {
  Trace trace;
  Device device;
  Resource resource;
  (*device.mutable_resources())[1] = resource;
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  std::vector<TraceEvent> complete_events(3);
  for (int i = 0; i < 3; ++i) {
    TraceEvent& ev = complete_events[i];
    ev.set_device_id(0);
    ev.set_resource_id(1);
    ev.set_name("Op");
    ev.set_timestamp_ps((i + 1) * 1000);
    ev.set_duration_ps(500);
    ev.set_group_id(10);
    if (i == 1) ev.set_serial(42);
    container.AddCompleteEvent(0, 1, &ev);
  }

  // Default options: sparse_event_metadata is false.
  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));
  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.complete_events_size(), 1);
  const xprof::TraceEventSeries& comp = response.complete_events(0);
  EXPECT_EQ(comp.event_metadata_size(), 3);
  EXPECT_EQ(comp.event_metadata_indices_size(), 0);
  EXPECT_EQ(comp.group_id_run_values_size(), 0);
  EXPECT_EQ(comp.group_id_run_lengths_size(), 0);
  EXPECT_EQ(comp.event_metadata(0).group_id(), 10);
  EXPECT_EQ(comp.event_metadata(1).serial(), 42u);
  EXPECT_EQ(comp.event_metadata(2).group_id(), 10);
}

TEST(DeltaSeriesProtoConverterTest,
     SortsProcessesAndThreadsDeterministicallyById) {
  Trace trace;
  Device device10;
  device10.set_name("Device 10");
  (*device10.mutable_resources())[30].set_name("Thread 30");
  (*device10.mutable_resources())[5].set_name("Thread 5");
  (*device10.mutable_resources())[20].set_name("Thread 20");
  (*trace.mutable_devices())[10] = device10;

  Device device2;
  device2.set_name("Device 2");
  (*device2.mutable_resources())[9].set_name("Thread 9");
  (*device2.mutable_resources())[1].set_name("Thread 1");
  (*trace.mutable_devices())[2] = device2;

  TestTraceEventsContainer container(trace);
  ASSERT_OK_AND_ASSIGN(std::string compressed_result,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           DeltaSeriesProtoConversionOptions{}, container));
  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.metadata().processes_size(), 2);
  EXPECT_EQ(response.metadata().processes(0).id(), 2u);
  ASSERT_EQ(response.metadata().processes(0).threads_size(), 2);
  EXPECT_EQ(response.metadata().processes(0).threads(0).id(), 1u);
  EXPECT_EQ(response.metadata().processes(0).threads(1).id(), 9u);

  EXPECT_EQ(response.metadata().processes(1).id(), 10u);
  ASSERT_EQ(response.metadata().processes(1).threads_size(), 3);
  EXPECT_EQ(response.metadata().processes(1).threads(0).id(), 5u);
  EXPECT_EQ(response.metadata().processes(1).threads(1).id(), 20u);
  EXPECT_EQ(response.metadata().processes(1).threads(2).id(), 30u);
}

TEST(DeltaSeriesProtoConverterTest,
     ExtractsHloModuleIdFromKernelDetailsViaRe2) {
  Trace trace;
  Device device;
  device.set_name("/device:GPU:0");
  (*device.mutable_resources())[1].set_name("Stream #0");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  auto make_gpu_event = [](uint64_t ts_ps, absl::string_view hlo_op,
                           absl::string_view hlo_module,
                           absl::string_view kernel_details) {
    TraceEvent ev;
    ev.set_device_id(0);
    ev.set_resource_id(1);
    ev.set_name("cutlass_gemm");
    ev.set_timestamp_ps(ts_ps);
    ev.set_duration_ps(100);
    RawData rd;
    TraceEventArguments::Argument* a1 = rd.mutable_args()->add_arg();
    a1->set_name("hlo_op");
    a1->set_str_value(std::string(hlo_op));
    TraceEventArguments::Argument* a2 = rd.mutable_args()->add_arg();
    a2->set_name("hlo_module");
    a2->set_str_value(std::string(hlo_module));
    TraceEventArguments::Argument* a3 = rd.mutable_args()->add_arg();
    a3->set_name("kernel_details");
    a3->set_str_value(std::string(kernel_details));
    ev.set_raw_data(rd.SerializeAsString());
    return ev;
  };

  TraceEvent ev1 =
      make_gpu_event(1000, "dot.1", "jit_step", "module:jit_step_88\nblock:1");
  TraceEvent ev2 =
      make_gpu_event(2000, "dot.2", "jit_eval", "module:jit_eval(99) grid:2");
  container.AddCompleteEvent(0, 1, &ev1);
  container.AddCompleteEvent(0, 1, &ev2);

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));

  ASSERT_EQ(response.complete_events_size(), 1);
  const xprof::TraceEventSeries& comp = response.complete_events(0);
  ASSERT_EQ(comp.event_metadata_size(), 2);
  EXPECT_EQ(response.interned_strings(comp.event_metadata(0).hlo_module_ref()),
            "jit_step(88)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(1).hlo_module_ref()),
            "jit_eval(99)");
}

TEST(DeltaSeriesProtoConverterTest,
     EmitsLengthPrefixedCompressedStreamFramesWithHeaderChunksAndTrailer) {
  Trace trace;
  trace.set_min_timestamp_ps(100);
  trace.set_max_timestamp_ps(5000);
  Device device;
  device.set_name("TPU 0");
  (*device.mutable_resources())[1].set_name("XLA Modules");
  (*device.mutable_resources())[2].set_name("XLA Ops");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  std::vector<TraceEvent> comp_events(5);
  for (int i = 0; i < 5; ++i) {
    TraceEvent& ev = comp_events[i];
    ev.set_device_id(0);
    ev.set_resource_id(2);
    ev.set_name("fusion_op");
    ev.set_timestamp_ps((i + 1) * 500);
    ev.set_duration_ps(100);
    ev.set_group_id(7);
    container.AddCompleteEvent(0, 2, &ev);
  }

  // Unpaired begin/end async events to exercise server-side P7 pairing.
  TraceEvent async_begin;
  async_begin.set_device_id(0);
  async_begin.set_name("AsyncCh");
  async_begin.set_timestamp_ps(1000);
  async_begin.set_duration_ps(0);
  async_begin.set_flow_id(0x100000005ULL);
  container.AddAsyncEvent(0, "AsyncCh", &async_begin);

  TraceEvent async_end;
  async_end.set_device_id(0);
  async_end.set_name("AsyncCh");
  async_end.set_timestamp_ps(1800);
  async_end.set_duration_ps(0);
  async_end.set_flow_id(0x100000005ULL);
  container.AddAsyncEvent(0, "AsyncCh", &async_end);

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  options.stream_max_events_per_segment = 2;

  ASSERT_OK_AND_ASSIGN(
      const std::string stream_bytes,
      ConvertTraceDataToCompressedStreamProto(options, container));

  std::vector<xprof::TraceStreamMessage> messages;
  size_t pos = 0;
  while (pos < stream_bytes.size()) {
    ASSERT_LE(pos + 4, stream_bytes.size());
    const uint32_t frame_len =
        static_cast<uint8_t>(stream_bytes[pos]) |
        (static_cast<uint32_t>(static_cast<uint8_t>(stream_bytes[pos + 1]))
         << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(stream_bytes[pos + 2]))
         << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(stream_bytes[pos + 3]))
         << 24);
    pos += 4;
    ASSERT_LE(pos + frame_len, stream_bytes.size());
    ASSERT_OK_AND_ASSIGN(const std::string decompressed,
                         ZstdCompression::Decompress(absl::string_view(
                             stream_bytes.data() + pos, frame_len)));
    pos += frame_len;
    xprof::TraceStreamMessage msg;
    ASSERT_TRUE(msg.ParseFromString(decompressed));
    messages.push_back(std::move(msg));
  }

  // Expect header, multiple chunk frames (5 events split by 2 -> 3 chunks),
  // and trailer.
  ASSERT_GE(messages.size(), 4u);
  EXPECT_TRUE(messages.front().has_header());
  EXPECT_TRUE(messages.back().has_trailer());

  const xprof::TraceStreamHeader& header = messages.front().header();
  ASSERT_EQ(header.tracks_size(), 2);
  EXPECT_EQ(header.tracks(0).kind(), xprof::TrackManifest::TRACK_KIND_COMPLETE);
  EXPECT_EQ(header.tracks(0).event_count(), 5u);
  // Post-pairing async count is 1 (paired from 2 begin/end instants).
  EXPECT_EQ(header.tracks(1).kind(), xprof::TrackManifest::TRACK_KIND_ASYNC);
  EXPECT_EQ(header.tracks(1).event_count(), 1u);

  const xprof::TraceStreamTrailer& trailer = messages.back().trailer();
  EXPECT_EQ(trailer.total_events(), 6u);
  ASSERT_EQ(trailer.track_event_counts_size(), 2);
  EXPECT_EQ(trailer.track_event_counts(0), 5u);
  EXPECT_EQ(trailer.track_event_counts(1), 1u);
}

TEST(DeltaSeriesProtoConverterTest,
     StreamProtoPropagatesCounterEventStatsRefOnFirstSegment) {
  Trace trace;
  Device device;
  (*trace.mutable_devices())[0] = device;

  std::vector<TraceEvent> cnt_events(3);
  for (int i = 0; i < 3; ++i) {
    TraceEvent& ev = cnt_events[i];
    ev.set_device_id(0);
    ev.set_name("HBM");
    ev.set_timestamp_ps((i + 1) * 1000);
    RawData rd;
    TraceEventArguments::Argument* arg = rd.mutable_args()->add_arg();
    arg->set_name("GiB");
    arg->set_double_value(1.5 * (i + 1));
    ev.set_raw_data(rd.SerializeAsString());
  }

  TestTraceEventsContainer container(trace);
  for (int i = 0; i < 3; ++i) {
    container.AddCounterEvent(0, "HBM", &cnt_events[i]);
  }

  DeltaSeriesProtoConversionOptions options;
  options.stream_max_events_per_segment = 2;
  ASSERT_OK_AND_ASSIGN(
      const std::string stream_bytes,
      ConvertTraceDataToCompressedStreamProto(options, container));

  std::vector<xprof::TraceStreamMessage> messages;
  size_t pos = 0;
  while (pos < stream_bytes.size()) {
    ASSERT_LE(pos + 4, stream_bytes.size());
    const uint32_t frame_len =
        static_cast<uint8_t>(stream_bytes[pos]) |
        (static_cast<uint32_t>(static_cast<uint8_t>(stream_bytes[pos + 1]))
         << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(stream_bytes[pos + 2]))
         << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(stream_bytes[pos + 3]))
         << 24);
    pos += 4;
    ASSERT_LE(pos + frame_len, stream_bytes.size());
    ASSERT_OK_AND_ASSIGN(const std::string decompressed,
                         ZstdCompression::Decompress(absl::string_view(
                             stream_bytes.data() + pos, frame_len)));
    pos += frame_len;
    xprof::TraceStreamMessage msg;
    ASSERT_TRUE(msg.ParseFromString(decompressed));
    messages.push_back(std::move(msg));
  }

  ASSERT_GE(messages.size(), 4u);
  const xprof::TraceStreamHeader& header = messages.front().header();
  ASSERT_EQ(header.tracks_size(), 1);
  EXPECT_EQ(header.tracks(0).kind(), xprof::TrackManifest::TRACK_KIND_COUNTER);

  std::vector<xprof::TrackSegment> counter_segments;
  for (size_t i = 1; i + 1 < messages.size(); ++i) {
    ASSERT_TRUE(messages[i].has_chunk());
    for (const xprof::TrackSegment& seg : messages[i].chunk().segments()) {
      counter_segments.push_back(seg);
    }
  }
  ASSERT_EQ(counter_segments.size(), 2u);
  EXPECT_EQ(counter_segments[0].first_event_ordinal(), 0u);
  ASSERT_TRUE(counter_segments[0].series().metadata().has_event_stats_ref());
  const uint32_t stats_ref =
      counter_segments[0].series().metadata().event_stats_ref();
  ASSERT_LT(stats_ref, static_cast<uint32_t>(header.interned_strings_size()));
  EXPECT_EQ(header.interned_strings(stats_ref), "GiB");
  EXPECT_FALSE(counter_segments[1].series().metadata().has_event_stats_ref());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
