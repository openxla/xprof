#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
