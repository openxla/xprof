#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <cstdint>
#include <map>
#include <string>

#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

class TestTraceEventsContainer {
 public:
  using RawDataType = RawData;

  explicit TestTraceEventsContainer(Trace* trace) : trace_(trace) {}

  void AddCounterEvent(uint32_t device_id, absl::string_view name,
                       TraceEvent* event) {
    events_by_device_[device_id]
        .counter_events_by_name[std::string(name)]
        .push_back(event);
  }

  void AddCompleteEvent(uint32_t device_id, uint64_t resource_id,
                        TraceEvent* event) {
    events_by_device_[device_id].events_by_resource[resource_id].push_back(
        event);
  }

  void AddAsyncEvent(uint32_t device_id, absl::string_view name,
                     TraceEvent* event) {
    events_by_device_[device_id]
        .counter_events_by_name[std::string(name)]
        .push_back(event);
  }

  const Trace& trace() const { return *trace_; }

  template <typename Callback>
  void ForAllTracks(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        if (!events.empty()) {
          callback(device_id, counter_name, events);
        }
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        if (!events.empty()) {
          callback(device_id, resource_id, events);
        }
      }
    }
  }

  template <typename Callback>
  void ForAllEvents(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        for (const auto& event : events) {
          callback(*event);
        }
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        for (const auto& event : events) {
          callback(*event);
        }
      }
    }
  }

 private:
  Trace* trace_;

  struct DeviceEvents {
    std::map<std::string, TraceEventTrack> counter_events_by_name;
    std::map<uint64_t, TraceEventTrack> events_by_resource;
  };
  std::map<uint32_t, DeviceEvents> events_by_device_;
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

  TestTraceEventsContainer container(&trace);
  container.AddCompleteEvent(0, 1, &event1);
  container.AddCompleteEvent(0, 1, &event2);

  StringOutput output;
  absl::Status status = ConvertTraceDataToCompressedDeltaSeriesProto(
      DeltaSeriesProtoConversionOptions{}, container, &output);
  ASSERT_TRUE(status.ok());

  // Decompress to verify the structure
  absl::StatusOr<std::string> decompressed =
      ZstdCompression::Decompress(output.str());
  ASSERT_TRUE(decompressed.ok());

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(*decompressed));

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

  TestTraceEventsContainer container(&trace);
  container.AddCounterEvent(0, "MyCounter", &event1);
  container.AddCounterEvent(0, "MyCounter", &event2);

  StringOutput output;
  absl::Status status = ConvertTraceDataToCompressedDeltaSeriesProto(
      DeltaSeriesProtoConversionOptions{}, container, &output);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<std::string> decompressed =
      ZstdCompression::Decompress(output.str());
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(*decompressed));

  ASSERT_EQ(response.counter_events_size(), 1);
  const auto& series = response.counter_events(0);

  EXPECT_EQ(series.metadata().process_id(), 0);

  ASSERT_EQ(series.deltas_size(), 2);
  EXPECT_EQ(series.deltas(0), 1000);
  EXPECT_EQ(series.deltas(1), 500);

  ASSERT_EQ(series.event_metadata_size(), 2);
  EXPECT_EQ(series.event_metadata(0).counter_value_uint64(), 1234);
  EXPECT_EQ(series.event_metadata(1).counter_value_uint64(), 5678);
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

  TestTraceEventsContainer container(&trace);
  container.AddAsyncEvent(0, "AsyncOp", &event1);
  container.AddAsyncEvent(0, "AsyncOp", &event2);

  StringOutput output;
  absl::Status status = ConvertTraceDataToCompressedDeltaSeriesProto(
      DeltaSeriesProtoConversionOptions{}, container, &output);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<std::string> decompressed =
      ZstdCompression::Decompress(output.str());
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(*decompressed));

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

  TestTraceEventsContainer container(&trace);

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

  StringOutput output;
  absl::Status status = ConvertTraceDataToCompressedDeltaSeriesProto(
      DeltaSeriesProtoConversionOptions{}, container, &output);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<std::string> decompressed =
      ZstdCompression::Decompress(output.str());
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(*decompressed));

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

  TestTraceEventsContainer container(&trace);
  container.AddCompleteEvent(0, 1, &event0);
  container.AddCompleteEvent(1, 1, &event1);

  StringOutput output;
  DeltaSeriesProtoConversionOptions options;
  options.mpmd_pipeline_view = true;
  absl::Status status =
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container, &output);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<std::string> decompressed =
      ZstdCompression::Decompress(output.str());
  ASSERT_TRUE(decompressed.ok());

  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(*decompressed));

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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
