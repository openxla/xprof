#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/io/coded_stream.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/delta_series/delta_series_converter_internal.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::EqualsProto;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
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

class ContainerWithEmptyTrack : public TestTraceEventsContainer {
 public:
  using TestTraceEventsContainer::TestTraceEventsContainer;

  template <typename Fn>
  void ForAllTracks(Fn&& fn) const {
    const TraceEventTrack empty_track;
    fn(0u, std::variant<uint64_t, absl::string_view>(999ULL), empty_track);
    TestTraceEventsContainer::ForAllTracks(std::forward<Fn>(fn));
  }
};

absl::StatusOr<xprof::TraceDataResponse> DecompressAndParseResponse(
    absl::string_view compressed) {
  absl::StatusOr<std::string> decompressed =
      ZstdCompression::Decompress(compressed);
  if (!decompressed.ok()) return decompressed.status();
  xprof::TraceDataResponse response;
  if (!response.ParseFromString(*decompressed)) {
    return absl::InternalError("Failed to parse TraceDataResponse.");
  }
  return response;
}


std::string MakeRawDataWithStrArgs(
    std::initializer_list<std::pair<absl::string_view, absl::string_view>>
        args) {
  RawData rd;
  for (const auto& [name, value] : args) {
    TraceEventArguments::Argument* arg = rd.mutable_args()->add_arg();
    arg->set_name(std::string(name));
    arg->set_str_value(std::string(value));
  }
  return rd.SerializeAsString();
}

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
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse full_response,
                       DecompressAndParseResponse(full_compressed));

  ASSERT_OK_AND_ASSIGN(std::string windowed_compressed,
                       ConvertTraceDataToCompressedDeltaSeriesProto(
                           options, windowed_container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse windowed_response,
                       DecompressAndParseResponse(windowed_compressed));

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

  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

TEST(DeltaSeriesStringTableTest, IncrementalHelpersAndMetadata) {
  DeltaSeriesStringTable string_table;
  EXPECT_EQ(string_table.InternString(""), 0);
  const uint32_t id1 = string_table.InternString("op_a");
  const uint32_t id2 = string_table.InternString("op_b");
  EXPECT_EQ(string_table.InternString("op_a"), id1);
  EXPECT_EQ(string_table.InternedString(id1), "op_a");
  EXPECT_EQ(string_table.InternedString(id2), "op_b");
  EXPECT_THAT(string_table.InternedString(999), IsEmpty());

  xprof::TraceDataResponse frame1;
  string_table.FlushInternedStringsIncrement(&frame1);
  ASSERT_EQ(frame1.interned_strings_size(), 3);
  EXPECT_EQ(frame1.interned_strings(0), "");
  EXPECT_EQ(frame1.interned_strings(1), "op_a");
  EXPECT_EQ(frame1.interned_strings(2), "op_b");

  xprof::TraceEventMetadata meta;
  const char* const kTfExecutorCategory = tsl::profiler::GetContextTypeString(
      tsl::profiler::ContextType::kTfExecutor);
  string_table.PopulateFlowAndGroupMetadata(/*flow_id=*/99, /*flow_category=*/2,
                                            /*group_id=*/7, &meta);
  EXPECT_EQ(meta.flow_id(), 99);
  EXPECT_EQ(meta.group_id(), 7);
  EXPECT_GT(meta.flow_category(), 0);
  EXPECT_EQ(string_table.InternedString(meta.flow_category()),
            kTfExecutorCategory);

  // Out-of-range uint32_t flow_category values (> 255) must not wrap around via
  // uint8_t truncation into a valid ContextType.
  xprof::TraceEventMetadata out_of_range_meta;
  string_table.PopulateFlowAndGroupMetadata(
      /*flow_id=*/100, /*flow_category=*/256 + 2, /*group_id=*/8,
      &out_of_range_meta);
  EXPECT_EQ(out_of_range_meta.flow_category(), 0u);

  // Generic (0) and Legacy (1) flow categories must not be interned.
  xprof::TraceEventMetadata generic_meta;
  string_table.PopulateFlowAndGroupMetadata(
      /*flow_id=*/101, /*flow_category=*/0, /*group_id=*/9, &generic_meta);
  EXPECT_EQ(generic_meta.flow_category(), 0u);

  xprof::TraceEventMetadata legacy_meta;
  string_table.PopulateFlowAndGroupMetadata(
      /*flow_id=*/102, /*flow_category=*/1, /*group_id=*/10, &legacy_meta);
  EXPECT_EQ(legacy_meta.flow_category(), 0u);

  // std::nullopt arguments must leave their fields at default 0.
  xprof::TraceEventMetadata empty_meta;
  string_table.PopulateFlowAndGroupMetadata(
      /*flow_id=*/std::nullopt, /*flow_category=*/std::nullopt,
      /*group_id=*/std::nullopt, &empty_meta);
  EXPECT_EQ(empty_meta.flow_id(), 0u);
  EXPECT_EQ(empty_meta.flow_category(), 0u);
  EXPECT_EQ(empty_meta.group_id(), 0);

  xprof::TraceDataResponse frame2;
  string_table.FlushInternedStringsIncrement(&frame2);
  EXPECT_THAT(frame2.interned_strings(), ElementsAre(kTfExecutorCategory));
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
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

  ASSERT_EQ(response.complete_events_size(), 1);
  const xprof::TraceEventSeries& comp = response.complete_events(0);
  ASSERT_EQ(comp.event_metadata_size(), 2);
  ASSERT_EQ(comp.event_metadata_indices_size(), 2);
  EXPECT_EQ(comp.event_metadata_indices(0), 2u);
  EXPECT_EQ(comp.event_metadata(0).flow_id(), 777u);
  EXPECT_EQ(comp.event_metadata(0).group_id(), 0u);
  EXPECT_EQ(comp.event_metadata_indices(1), 4u);
  EXPECT_EQ(comp.event_metadata(1).serial(), 99u);
  EXPECT_EQ(comp.event_metadata(1).group_id(), 0u);

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
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

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

TEST(DeltaSeriesProtoConverterTest, ExtractsHloModuleIdFromKernelDetails) {
  Trace trace;
  Device device;
  device.set_name("/device:GPU:0");
  (*device.mutable_resources())[1].set_name("Stream #0");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  auto make_gpu_event = [](uint64_t ts_ps, absl::string_view hlo_op,
                           absl::string_view hlo_module,
                           absl::string_view kernel_details = "",
                           absl::string_view hlo_module_id = "",
                           absl::string_view program_id = "") {
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
    if (!kernel_details.empty()) {
      TraceEventArguments::Argument* a3 = rd.mutable_args()->add_arg();
      a3->set_name("kernel_details");
      a3->set_str_value(std::string(kernel_details));
    }
    if (!hlo_module_id.empty()) {
      TraceEventArguments::Argument* a4 = rd.mutable_args()->add_arg();
      a4->set_name("hlo_module_id");
      a4->set_str_value(std::string(hlo_module_id));
    }
    if (!program_id.empty()) {
      TraceEventArguments::Argument* a5 = rd.mutable_args()->add_arg();
      a5->set_name("program_id");
      a5->set_str_value(std::string(program_id));
    }
    ev.set_raw_data(rd.SerializeAsString());
    return ev;
  };

  TraceEvent ev1 = make_gpu_event(
      1000, "dot.1", "jit_step",
      "grid:(128,1,1) block:(64,1,1) module:jit_step_88\nblock:1");
  TraceEvent ev2 = make_gpu_event(
      2000, "dot.2", "jit_eval", "launch:(1,2) module:jit_eval(99) grid:(4,4)");
  TraceEvent ev3 = make_gpu_event(3000, "dot.3", "jit_step(88)",
                                  "module:jit_step_88\nblock:1");
  TraceEvent ev4 = make_gpu_event(4000, "dot.4", "jit_step(88)", "", "88", "");
  TraceEvent ev5 = make_gpu_event(5000, "dot.5", "jit_eval(99)", "", "", "99");
  // Edge cases: trailing "module:" at EOF, tab/CR delimiters, trailing
  // underscore, non-digits below '0' ('/') and above '9' (':'), leading '(',
  // unclosed '(', empty parens, and multi-token fallback.
  TraceEvent ev6 = make_gpu_event(6000, "dot.6", "m_eof", "module:");
  TraceEvent ev7 =
      make_gpu_event(7000, "dot.7", "m_tab", "module:bad_\tmodule:ok_12\r");
  TraceEvent ev8 = make_gpu_event(8000, "dot.8", "m_nondigit",
                                  "module:a_/9 module:b_:9 module:(12)");
  TraceEvent ev9 =
      make_gpu_event(9000, "dot.9", "m_parens", "module:x(1 module:y()");
  container.AddCompleteEvent(0, 1, &ev1);
  container.AddCompleteEvent(0, 1, &ev2);
  container.AddCompleteEvent(0, 1, &ev3);
  container.AddCompleteEvent(0, 1, &ev4);
  container.AddCompleteEvent(0, 1, &ev5);
  container.AddCompleteEvent(0, 1, &ev6);
  container.AddCompleteEvent(0, 1, &ev7);
  container.AddCompleteEvent(0, 1, &ev8);
  container.AddCompleteEvent(0, 1, &ev9);

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

  ASSERT_EQ(response.complete_events_size(), 1);
  const xprof::TraceEventSeries& comp = response.complete_events(0);
  ASSERT_EQ(comp.event_metadata_size(), 9);
  EXPECT_EQ(response.interned_strings(comp.event_metadata(0).hlo_module_ref()),
            "jit_step(88)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(1).hlo_module_ref()),
            "jit_eval(99)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(2).hlo_module_ref()),
            "jit_step(88)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(3).hlo_module_ref()),
            "jit_step(88)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(4).hlo_module_ref()),
            "jit_eval(99)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(5).hlo_module_ref()),
            "m_eof");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(6).hlo_module_ref()),
            "m_tab(12)");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(7).hlo_module_ref()),
            "m_nondigit");
  EXPECT_EQ(response.interned_strings(comp.event_metadata(8).hlo_module_ref()),
            "m_parens");
}

TEST(DeltaSeriesProtoConverterTest,
     XlaOpsLeadingNoRawDataStillSuppressesTpuRedundantMetadata) {
  Trace trace;
  Device device;
  device.set_name("/device:TPU:0");
  (*device.mutable_resources())[1].set_name("XLA Ops");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);
  std::vector<TraceEvent> events(70);
  for (size_t i = 0; i < events.size(); ++i) {
    TraceEvent& ev = events[i];
    ev.set_device_id(0);
    ev.set_resource_id(1);
    ev.set_name(absl::StrCat("fusion.", i));
    ev.set_timestamp_ps(1000ULL + i * 100ULL);
    ev.set_duration_ps(50);
    if (i > 0) {
      RawData rd;
      TraceEventArguments::Argument* a = rd.mutable_args()->add_arg();
      a->set_name("program_id");
      a->set_uint_value(7);
      ev.set_raw_data(rd.SerializeAsString());
    }
    container.AddCompleteEvent(0, 1, &ev);
  }

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed_mono,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse mono_resp,
                       DecompressAndParseResponse(compressed_mono));
  ASSERT_EQ(mono_resp.complete_events_size(), 1);
  EXPECT_EQ(mono_resp.complete_events(0).event_metadata_size(), 0);
}

TEST(DeltaSeriesProtoConverterTest,
     HeaderFirstWireOrderingAndChunkedCompleteEvents) {
  constexpr size_t kNumEvents = 200000;
  Trace trace;
  Device device;
  Resource resource;
  resource.set_name("Compute Stream");
  (*device.mutable_resources())[1] = resource;
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);

  std::vector<TraceEvent> events(kNumEvents);
  for (size_t i = 0; i < kNumEvents; ++i) {
    TraceEvent& ev = events[i];
    ev.set_device_id(0);
    ev.set_resource_id(1);
    ev.set_name("FusedOp");
    ev.set_timestamp_ps(static_cast<uint64_t>(i) * 1000 + 42);
    ev.set_duration_ps(static_cast<uint64_t>((i % 500) + 1));
    container.AddCompleteEvent(0, 1, &ev);
  }

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(std::string decompressed,
                       ZstdCompression::Decompress(compressed_result));

  // (a) Verify raw decompressed wire order: header fields appear BEFORE
  // Field 1.
  tsl::protobuf::io::ArrayInputStream array_stream(decompressed.data(),
                                                   decompressed.size());
  tsl::protobuf::io::CodedInputStream coded_stream(&array_stream);
  uint32_t tag;
  bool seen_complete_events = false;
  bool seen_header_fields = false;
  while ((tag = coded_stream.ReadTag()) != 0) {
    const uint32_t field_number = tag >> 3;
    if (field_number == 1) {
      seen_complete_events = true;
      EXPECT_TRUE(seen_header_fields)
          << "Complete events (Field 1) appeared before header fields!";
    } else {
      EXPECT_FALSE(seen_complete_events)
          << "Header field " << field_number << " appeared after Field 1!";
      seen_header_fields = true;
    }
    const uint32_t wire_type = tag & 7;
    if (wire_type == 0) {
      uint64_t val;
      ASSERT_TRUE(coded_stream.ReadVarint64(&val));
    } else if (wire_type == 2) {
      uint32_t length;
      ASSERT_TRUE(coded_stream.ReadVarint32(&length));
      ASSERT_TRUE(coded_stream.Skip(length));
    } else if (wire_type == 1) {
      ASSERT_TRUE(coded_stream.Skip(8));
    } else if (wire_type == 5) {
      ASSERT_TRUE(coded_stream.Skip(4));
    } else {
      FAIL() << "Unexpected wire type: " << wire_type;
    }
  }
  EXPECT_TRUE(seen_header_fields);
  EXPECT_TRUE(seen_complete_events);

  // (b) Verify parsing and chunking: Field 1 is split into 4 chunks
  // (65536+65536+65536+3392).
  xprof::TraceDataResponse response;
  ASSERT_TRUE(response.ParseFromString(decompressed));
  ASSERT_EQ(response.complete_events_size(), 4);
  EXPECT_EQ(response.complete_events(0).deltas_size(), 65536);
  EXPECT_EQ(response.complete_events(1).deltas_size(), 65536);
  EXPECT_EQ(response.complete_events(2).deltas_size(), 65536);
  EXPECT_EQ(response.complete_events(3).deltas_size(), 3392);

  // (c) Verify 100% 1 ps lossless timestamp and duration precision across all
  // 200,000 events.
  size_t event_idx = 0;
  for (int chunk_idx = 0; chunk_idx < response.complete_events_size();
       ++chunk_idx) {
    const xprof::TraceEventSeries& chunk = response.complete_events(chunk_idx);
    EXPECT_LE(chunk.deltas_size(), 65536);
    EXPECT_EQ(chunk.deltas_size(), chunk.durations_size());

    uint64_t current_ts = 0;
    for (int j = 0; j < chunk.deltas_size(); ++j) {
      current_ts += chunk.deltas(j);
      const uint64_t expected_ts = static_cast<uint64_t>(event_idx) * 1000 + 42;
      const uint64_t expected_dur =
          static_cast<uint64_t>((event_idx % 500) + 1);
      ASSERT_EQ(current_ts, expected_ts)
          << "Timestamp mismatch at event index " << event_idx;
      ASSERT_EQ(chunk.durations(j), expected_dur)
          << "Duration mismatch at event index " << event_idx;
      ++event_idx;
    }
  }
  EXPECT_EQ(event_idx, kNumEvents);
}

TEST(DeltaSeriesProtoConverterTest,
     SparseMetadataExtractionArgTypesProbeAndAsyncEdgeCases) {
  Trace trace;
  (*trace.mutable_name_table())[100] = "RefEventName";
  (*trace.mutable_name_table())[200] = "ref_hlo_op";
  Device device;
  device.set_name("/device:TPU:0");
  (*device.mutable_resources())[1].set_name("Compute Utilization");
  (*device.mutable_resources())[2].set_name("Regular Thread");
  (*device.mutable_resources())[10].set_name("XLA Ops");
  (*device.mutable_resources())[11].set_name("XLA Ops");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);
  std::vector<TraceEvent> events;
  events.reserve(160);

  auto make_raw =
      [](std::initializer_list<std::pair<std::string, std::string>> str_args,
         uint64_t ref_arg_val = 0, int64_t int_arg_val = 0,
         uint64_t uint_arg_val = 0, bool set_numeric = false) {
        tensorflow::profiler::RawData raw;
        TraceEventArguments* args = raw.mutable_args();
        for (const auto& [k, v] : str_args) {
          TraceEventArguments::Argument* a = args->add_arg();
          a->set_name(k);
          a->set_str_value(v);
        }
        if (ref_arg_val != 0) {
          TraceEventArguments::Argument* a = args->add_arg();
          a->set_name("hlo_op");
          a->set_ref_value(ref_arg_val);
        }
        if (set_numeric) {
          TraceEventArguments::Argument* a1 = args->add_arg();
          a1->set_name("program_id");
          a1->set_int_value(int_arg_val);
          TraceEventArguments::Argument* a2 = args->add_arg();
          a2->set_name("hlo_module_id");
          a2->set_uint_value(uint_arg_val);
        }
        return raw.SerializeAsString();
      };

  // Compute Utilization events: Name arg, hlo_op fallback, corrupt raw_data,
  // empty-args raw_data, bad kernel_details, and ref_value/int/uint args.
  {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(1);
    e.set_name_ref(100);
    e.set_timestamp_ps(1000);
    e.set_duration_ps(100);
    e.set_raw_data(make_raw({{"Name", "util_name_op"},
                             {"hlo_module", "plain_mod"},
                             {"kernel_details", "hlo_module: bad_no_paren "}}));
    container.AddCompleteEvent(0, 1, &e);
  }
  {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(1);
    e.set_name_ref(999);  // Missing in name_table -> ""
    e.set_timestamp_ps(2000);
    e.set_duration_ps(100);
    e.set_raw_data(make_raw({{"hlo_module", "mod_num"}, {"Name", ""}},
                            /*ref_arg_val=*/200, /*int_arg_val=*/7,
                            /*uint_arg_val=*/8, /*set_numeric=*/true));
    container.AddCompleteEvent(0, 1, &e);
  }
  {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(1);
    e.set_name("corrupt_raw");
    e.set_timestamp_ps(3000);
    e.set_duration_ps(100);
    e.set_raw_data("\xff\xff\xff");
    container.AddCompleteEvent(0, 1, &e);
  }
  {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(1);
    e.set_name("no_args_raw");
    e.set_timestamp_ps(4000);
    e.set_duration_ps(100);
    tensorflow::profiler::RawData empty_raw;
    e.set_raw_data(empty_raw.SerializeAsString());
    container.AddCompleteEvent(0, 1, &e);
  }
  // Regular thread: missing hlo_module (returns false), explicit group_id = -1
  // (must stay in RLE without wrapping uint32 metadata.group_id), and
  // non-generic flow_category.
  {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(2);
    e.set_name("regular_op");
    e.set_timestamp_ps(5000);
    e.set_duration_ps(100);
    e.set_group_id(-1);
    e.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kTfExecutor));
    e.set_raw_data(make_raw({{"hlo_op", "only_op"}}));
    container.AddCompleteEvent(0, 2, &e);
  }
  // XLA Ops thread 10: 66 events with no raw_data emit no sparse metadata.
  for (int i = 0; i < 66; ++i) {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(10);
    e.set_name("xla_no_raw");
    e.set_timestamp_ps(10000 + i * 100);
    e.set_duration_ps(50);
    container.AddCompleteEvent(0, 10, &e);
  }
  // XLA Ops thread 11: 66 events where only event 0 carries HLO raw_data.
  for (int i = 0; i < 66; ++i) {
    TraceEvent& e = events.emplace_back();
    e.set_device_id(0);
    e.set_resource_id(11);
    e.set_name("xla_with_raw");
    e.set_timestamp_ps(20000 + i * 100);
    e.set_duration_ps(50);
    if (i == 0) {
      e.set_raw_data(make_raw({{"hlo_op", "other_op"}, {"hlo_module", "m1"}}));
    }
    container.AddCompleteEvent(0, 11, &e);
  }
  // Async events: group_id RLE (same + changed), non-generic flow_category,
  // and HLO metadata refs.
  std::vector<TraceEvent> async_events(3);
  for (int i = 0; i < 3; ++i) {
    TraceEvent& e = async_events[i];
    e.set_device_id(0);
    e.set_name("async_op");
    e.set_timestamp_ps(30000 + i * 100);
    e.set_duration_ps(50);
    e.set_flow_id(500 + i);
    e.set_group_id(i < 2 ? 10 : 20);
    if (i == 1) {
      e.set_flow_category(
          static_cast<uint32_t>(tsl::profiler::ContextType::kTfExecutor));
      e.set_raw_data(make_raw({{"hlo_op", "async_hlo"}, {"hlo_module", "am"}}));
    }
    container.AddAsyncEvent(0, "async_op", &e);
  }

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      std::string compressed_result,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed_result));

  auto get_interned = [&response](int64_t ref) -> absl::string_view {
    if (ref < 0 || ref >= response.interned_strings_size()) {
      ADD_FAILURE() << "Out-of-bounds interned string ref: " << ref;
      return "";
    }
    return response.interned_strings(static_cast<int>(ref));
  };
  const char* const kTfExecutorCategory = tsl::profiler::GetContextTypeString(
      tsl::profiler::ContextType::kTfExecutor);

  ASSERT_EQ(response.complete_events_size(), 4);

  // Thread 1 (Compute Utilization): Name arg, ref_value hlo_op fallback with
  // uint_value hlo_module_id precedence over int_value program_id, missing
  // name_ref=999, and corrupt/empty raw_data skipped.
  const xprof::TraceEventSeries& comp_util = response.complete_events(0);
  EXPECT_EQ(comp_util.metadata().thread_id(), 1u);
  ASSERT_EQ(comp_util.deltas_size(), 4);
  ASSERT_EQ(comp_util.name_refs_size(), 4);
  EXPECT_EQ(get_interned(comp_util.name_refs(0)), "RefEventName");
  EXPECT_EQ(get_interned(comp_util.name_refs(1)), "");
  EXPECT_EQ(get_interned(comp_util.name_refs(2)), "corrupt_raw");
  EXPECT_EQ(get_interned(comp_util.name_refs(3)), "no_args_raw");
  EXPECT_THAT(comp_util.group_id_run_values(), ElementsAre(-1));
  EXPECT_THAT(comp_util.group_id_run_lengths(), ElementsAre(4));
  EXPECT_THAT(comp_util.event_metadata_indices(), ElementsAre(0, 1));
  ASSERT_EQ(comp_util.event_metadata_size(), 2);
  EXPECT_EQ(get_interned(comp_util.event_metadata(0).hlo_op_ref()),
            "util_name_op");
  EXPECT_EQ(get_interned(comp_util.event_metadata(0).hlo_module_ref()),
            "plain_mod");
  EXPECT_EQ(comp_util.event_metadata(0).group_id(), 0u);
  EXPECT_EQ(get_interned(comp_util.event_metadata(1).hlo_op_ref()),
            "ref_hlo_op");
  EXPECT_EQ(get_interned(comp_util.event_metadata(1).hlo_module_ref()),
            "mod_num(8)");
  EXPECT_EQ(comp_util.event_metadata(1).group_id(), 0u);

  // Thread 2 (Regular Thread): partial HLO args omitted; non-generic
  // flow_category emitted; explicit group_id=-1 stays only in RLE.
  const xprof::TraceEventSeries& regular = response.complete_events(1);
  EXPECT_EQ(regular.metadata().thread_id(), 2u);
  ASSERT_EQ(regular.deltas_size(), 1);
  EXPECT_THAT(regular.group_id_run_values(), ElementsAre(-1));
  EXPECT_THAT(regular.group_id_run_lengths(), ElementsAre(1));
  EXPECT_THAT(regular.event_metadata_indices(), ElementsAre(0));
  ASSERT_EQ(regular.event_metadata_size(), 1);
  EXPECT_EQ(get_interned(regular.event_metadata(0).flow_category()),
            kTfExecutorCategory);
  EXPECT_FALSE(regular.event_metadata(0).has_hlo_op_ref());
  EXPECT_FALSE(regular.event_metadata(0).has_hlo_module_ref());
  EXPECT_EQ(regular.event_metadata(0).flow_id(), 0u);
  EXPECT_EQ(regular.event_metadata(0).group_id(), 0u);

  // Thread 10 (XLA Ops without raw_data): no sparse metadata emitted.
  const xprof::TraceEventSeries& xla_no_raw = response.complete_events(2);
  EXPECT_EQ(xla_no_raw.metadata().thread_id(), 10u);
  EXPECT_EQ(xla_no_raw.deltas_size(), 66);
  EXPECT_THAT(xla_no_raw.event_metadata_indices(), IsEmpty());
  EXPECT_THAT(xla_no_raw.event_metadata(), IsEmpty());
  EXPECT_THAT(xla_no_raw.group_id_run_values(), ElementsAre(-1));
  EXPECT_THAT(xla_no_raw.group_id_run_lengths(), ElementsAre(66));

  // Thread 11 (XLA Ops with raw_data on event 0): single sparse metadata entry.
  const xprof::TraceEventSeries& xla_with_raw = response.complete_events(3);
  EXPECT_EQ(xla_with_raw.metadata().thread_id(), 11u);
  EXPECT_EQ(xla_with_raw.deltas_size(), 66);
  EXPECT_THAT(xla_with_raw.event_metadata_indices(), ElementsAre(0));
  ASSERT_EQ(xla_with_raw.event_metadata_size(), 1);
  EXPECT_EQ(get_interned(xla_with_raw.event_metadata(0).hlo_op_ref()),
            "other_op");
  EXPECT_EQ(get_interned(xla_with_raw.event_metadata(0).hlo_module_ref()),
            "m1");
  EXPECT_EQ(xla_with_raw.event_metadata(0).group_id(), 0u);
  EXPECT_THAT(xla_with_raw.group_id_run_values(), ElementsAre(-1));
  EXPECT_THAT(xla_with_raw.group_id_run_lengths(), ElementsAre(66));

  ASSERT_EQ(response.async_events_size(), 1);
  const xprof::TraceEventSeries& async_series = response.async_events(0);
  EXPECT_THAT(async_series.group_id_run_values(), ElementsAre(10, 20));
  EXPECT_THAT(async_series.group_id_run_lengths(), ElementsAre(2, 1));
  EXPECT_THAT(async_series.event_metadata_indices(), ElementsAre(1));
  ASSERT_EQ(async_series.event_metadata_size(), 1);
  EXPECT_EQ(async_series.event_metadata(0).flow_id(), 0u);
  EXPECT_EQ(async_series.event_metadata(0).group_id(), 0u);
  EXPECT_EQ(get_interned(async_series.event_metadata(0).flow_category()),
            kTfExecutorCategory);
  EXPECT_EQ(get_interned(async_series.event_metadata(0).hlo_op_ref()),
            "async_hlo");
  EXPECT_EQ(get_interned(async_series.event_metadata(0).hlo_module_ref()),
            "am");
}

TEST(DeltaSeriesProtoConverterTest,
     SerializeTraceDataResponseHeaderFirstArenaAllocated) {
  google::protobuf::Arena arena;
  xprof::TraceDataResponse* response =
      google::protobuf::Arena::Create<xprof::TraceDataResponse>(&arena);
  ASSERT_EQ(response->GetArena(), &arena);

  response->add_interned_strings("");
  response->add_interned_strings("arena_op");
  response->set_full_timespan_start_ps(1000);
  response->set_full_timespan_end_ps(5000);
  xprof::Process* proc = response->mutable_metadata()->add_processes();
  proc->set_id(7);
  proc->set_name("ArenaDevice");
  xprof::Thread* thr = proc->add_threads();
  thr->set_id(3);
  thr->set_name("ArenaThread");

  for (int c = 0; c < 3; ++c) {
    xprof::TraceEventSeries* series = response->add_complete_events();
    series->mutable_metadata()->set_process_id(7);
    series->mutable_metadata()->set_thread_id(3);
    series->add_deltas(static_cast<uint64_t>(1000 + c * 500));
    series->add_deltas(200);
    series->add_durations(100);
    series->add_durations(150);
    series->add_name_refs(1);
    series->add_name_refs(1);
  }

  ZstdCompressorContext zstd_ctx(/*compression_level=*/1);
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed_with_ctx,
      SerializeTraceDataResponseHeaderFirst(*response, &zstd_ctx));
  ASSERT_OK_AND_ASSIGN(const std::string compressed_default,
                       SerializeTraceDataResponseHeaderFirst(*response));
  EXPECT_EQ(compressed_with_ctx, compressed_default);

  // Verify the const-read response remains intact on the same Arena.
  ASSERT_EQ(response->complete_events_size(), 3);
  for (int c = 0; c < 3; ++c) {
    EXPECT_EQ(response->complete_events(c).GetArena(), &arena);
    EXPECT_EQ(response->complete_events(c).deltas_size(), 2);
  }

  // Verify header-first wire ordering and lossless round-trip.
  ASSERT_OK_AND_ASSIGN(const std::string decompressed,
                       ZstdCompression::Decompress(compressed_with_ctx));
  tsl::protobuf::io::ArrayInputStream array_stream(decompressed.data(),
                                                   decompressed.size());
  tsl::protobuf::io::CodedInputStream coded_stream(&array_stream);
  uint32_t tag = 0;
  bool seen_complete_events = false;
  bool seen_header_fields = false;
  while ((tag = coded_stream.ReadTag()) != 0) {
    const uint32_t field_number = tag >> 3;
    if (field_number == 1) {
      seen_complete_events = true;
      EXPECT_TRUE(seen_header_fields);
    } else {
      EXPECT_FALSE(seen_complete_events);
      seen_header_fields = true;
    }
    const uint32_t wire_type = tag & 7;
    if (wire_type == 0) {
      uint64_t val = 0;
      ASSERT_TRUE(coded_stream.ReadVarint64(&val));
    } else if (wire_type == 2) {
      uint32_t length = 0;
      ASSERT_TRUE(coded_stream.ReadVarint32(&length));
      ASSERT_TRUE(coded_stream.Skip(length));
    } else {
      FAIL() << "Unexpected wire type: " << wire_type;
    }
  }
  EXPECT_TRUE(seen_header_fields);
  EXPECT_TRUE(seen_complete_events);

  xprof::TraceDataResponse roundtrip;
  ASSERT_TRUE(roundtrip.ParseFromString(decompressed));
  EXPECT_THAT(roundtrip, EqualsProto(*response));
}

TEST(DeltaSeriesProtoConverterTest,
     ExtractHloStringsShortCircuitAndTrackModesParity) {
  Trace trace;
  (*trace.mutable_name_table())[10] = "ref_op_name";
  (*trace.mutable_name_table())[20] = "ref_mod_name";
  Device dev;
  dev.set_name("/device:TPU:0");
  (*dev.mutable_resources())[1].set_name("Regular Stream");
  (*dev.mutable_resources())[2].set_name("XLA Ops");
  (*dev.mutable_resources())[3].set_name("Data Motion Layers Utilization");
  (*trace.mutable_devices())[0] = dev;

  TestTraceEventsContainer container(trace);
  std::vector<TraceEvent> events;
  events.reserve(16);

  auto add_event = [&](uint64_t tid, absl::string_view name, uint64_t ts_ps,
                       const RawData& rd) {
    TraceEvent& ev = events.emplace_back();
    ev.set_device_id(0);
    ev.set_resource_id(tid);
    ev.set_name(std::string(name));
    ev.set_timestamp_ps(ts_ps);
    ev.set_duration_ps(100);
    ev.set_raw_data(rd.SerializeAsString());
    container.AddCompleteEvent(0, tid, &ev);
  };

  // Regular thread:
  // 0) Non-HLO raw_data -> short-circuit skip.
  {
    RawData rd;
    TraceEventArguments::Argument* a = rd.mutable_args()->add_arg();
    a->set_name("shape");
    a->set_str_value("f32[128,128]");
    add_event(1, "reg_no_hlo", 1000, rd);
  }
  // 1) Only hlo_op (missing hlo_module) -> short-circuit skip.
  {
    RawData rd;
    TraceEventArguments::Argument* a = rd.mutable_args()->add_arg();
    a->set_name("hlo_op");
    a->set_str_value("only_op");
    add_event(1, "reg_only_op", 2000, rd);
  }
  // 2) Only hlo_module (missing hlo_op) -> short-circuit skip.
  {
    RawData rd;
    TraceEventArguments::Argument* a = rd.mutable_args()->add_arg();
    a->set_name("hlo_module");
    a->set_str_value("only_mod");
    add_event(1, "reg_only_mod", 3000, rd);
  }
  // 3) Both hlo_op (ref_value) and hlo_module (ref_value) + program_id (uint) +
  //    unrelated args (double_value, Name) -> extracted cleanly.
  {
    RawData rd;
    TraceEventArguments::Argument* unrelated = rd.mutable_args()->add_arg();
    unrelated->set_name("flops");
    unrelated->set_double_value(99.5);
    TraceEventArguments::Argument* ignored_name = rd.mutable_args()->add_arg();
    ignored_name->set_name("Name");
    ignored_name->set_str_value("should_be_ignored_on_regular");
    TraceEventArguments::Argument* op = rd.mutable_args()->add_arg();
    op->set_name("hlo_op");
    op->set_ref_value(10);
    TraceEventArguments::Argument* mod = rd.mutable_args()->add_arg();
    mod->set_name("hlo_module");
    mod->set_ref_value(20);
    TraceEventArguments::Argument* pid = rd.mutable_args()->add_arg();
    pid->set_name("program_id");
    pid->set_uint_value(42);
    add_event(1, "reg_full_hlo", 4000, rd);
  }

  // XLA Ops thread:
  // 0) hlo_op equals event_name, hlo_module present -> op omitted, module kept.
  {
    RawData rd;
    TraceEventArguments::Argument* op = rd.mutable_args()->add_arg();
    op->set_name("hlo_op");
    op->set_str_value("fusion.1");
    TraceEventArguments::Argument* mod = rd.mutable_args()->add_arg();
    mod->set_name("hlo_module");
    mod->set_str_value("train_step");
    TraceEventArguments::Argument* mid = rd.mutable_args()->add_arg();
    mid->set_name("hlo_module_id");
    mid->set_int_value(77);
    add_event(2, "fusion.1", 5000, rd);
  }
  // 1) hlo_op differs from event_name, no hlo_module -> op kept.
  {
    RawData rd;
    TraceEventArguments::Argument* op = rd.mutable_args()->add_arg();
    op->set_name("hlo_op");
    op->set_str_value("custom_call.9");
    add_event(2, "wrapper_op", 6000, rd);
  }

  // Data Motion Layers thread:
  // 0) Name takes precedence over hlo_op.
  {
    RawData rd;
    TraceEventArguments::Argument* op = rd.mutable_args()->add_arg();
    op->set_name("hlo_op");
    op->set_str_value("fallback_op");
    TraceEventArguments::Argument* nm = rd.mutable_args()->add_arg();
    nm->set_name("Name");
    nm->set_str_value("dml_preferred_name");
    add_event(3, "dml_ev0", 7000, rd);
  }
  // 1) Empty Name falls back to hlo_op.
  {
    RawData rd;
    TraceEventArguments::Argument* nm = rd.mutable_args()->add_arg();
    nm->set_name("Name");
    nm->set_str_value("");
    TraceEventArguments::Argument* op = rd.mutable_args()->add_arg();
    op->set_name("hlo_op");
    op->set_str_value("dml_fallback_op");
    add_event(3, "dml_ev1", 8000, rd);
  }

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed_mono,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse resp,
                       DecompressAndParseResponse(compressed_mono));

  auto find_series = [&](uint64_t tid) -> const xprof::TraceEventSeries* {
    for (const xprof::TraceEventSeries& s : resp.complete_events()) {
      if (s.metadata().thread_id() == tid) return &s;
    }
    return nullptr;
  };

  const xprof::TraceEventSeries* reg = find_series(1);
  ASSERT_NE(reg, nullptr);
  ASSERT_EQ(reg->event_metadata_size(), 1);
  EXPECT_THAT(reg->event_metadata_indices(), ElementsAre(3));
  EXPECT_EQ(resp.interned_strings(reg->event_metadata(0).hlo_op_ref()),
            "ref_op_name");
  EXPECT_EQ(resp.interned_strings(reg->event_metadata(0).hlo_module_ref()),
            "ref_mod_name(42)");

  const xprof::TraceEventSeries* xla = find_series(2);
  ASSERT_NE(xla, nullptr);
  ASSERT_EQ(xla->event_metadata_size(), 2);
  EXPECT_THAT(xla->event_metadata_indices(), ElementsAre(0, 1));
  EXPECT_FALSE(xla->event_metadata(0).has_hlo_op_ref());
  EXPECT_EQ(resp.interned_strings(xla->event_metadata(0).hlo_module_ref()),
            "train_step(77)");
  EXPECT_EQ(resp.interned_strings(xla->event_metadata(1).hlo_op_ref()),
            "custom_call.9");
  EXPECT_FALSE(xla->event_metadata(1).has_hlo_module_ref());

  const xprof::TraceEventSeries* dml = find_series(3);
  ASSERT_NE(dml, nullptr);
  ASSERT_EQ(dml->event_metadata_size(), 2);
  EXPECT_THAT(dml->event_metadata_indices(), ElementsAre(0, 1));
  EXPECT_EQ(resp.interned_strings(dml->event_metadata(0).hlo_op_ref()),
            "dml_preferred_name");
  EXPECT_EQ(resp.interned_strings(dml->event_metadata(1).hlo_op_ref()),
            "dml_fallback_op");
}

TEST(DeltaSeriesStringTableTest, NonCopyableAndMovable) {
  static_assert(!std::is_copy_constructible_v<DeltaSeriesStringTable>);
  static_assert(!std::is_copy_assignable_v<DeltaSeriesStringTable>);
  static_assert(std::is_move_constructible_v<DeltaSeriesStringTable>);
  static_assert(std::is_move_assignable_v<DeltaSeriesStringTable>);

  DeltaSeriesStringTable table;
  const uint32_t id = table.InternString("moved_string");
  DeltaSeriesStringTable moved(std::move(table));
  EXPECT_EQ(moved.InternedString(id), "moved_string");
  EXPECT_EQ(moved.InternString("moved_string"), id);
}

TEST(DeltaSeriesProtoConverterTest,
     GetTraceMetadataResolves64BitResourceMapKeys) {
  // Verifies that GetTraceMetadata() resolves resource map keys above 2^32
  // without dropping them, while Thread.id currently remains a 32-bit uint32
  // field in TraceDataResponse.
  Trace trace;
  Device device;
  device.set_name("/device:GPU:0");
  constexpr uint64_t kHighResourceId = (1ULL << 33) + 7ULL;
  (*device.mutable_resources())[1].set_name("LowThread");
  (*device.mutable_resources())[kHighResourceId].set_name("High64BitThread");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed,
      ConvertTraceDataToCompressedDeltaSeriesProto({}, container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed));
  ASSERT_EQ(response.metadata().processes_size(), 1);
  ASSERT_EQ(response.metadata().processes(0).threads_size(), 2);
  EXPECT_EQ(response.metadata().processes(0).threads(0).id(), 1u);
  EXPECT_EQ(response.metadata().processes(0).threads(0).name(), "LowThread");
  EXPECT_EQ(response.metadata().processes(0).threads(1).id(),
            static_cast<uint32_t>(kHighResourceId));
  EXPECT_EQ(response.metadata().processes(0).threads(1).name(),
            "High64BitThread");
}

TEST(DeltaSeriesProtoConverterTest, ProtobufMessageSizeGuards) {
  EXPECT_THAT(internal::ValidateProtobufMessageSize(
                  internal::kMaxProtobufSerializedBytes, "test"),
              IsOk());
  EXPECT_THAT(internal::ValidateProtobufMessageSize(
                  internal::kMaxProtobufSerializedBytes + 1, "test"),
              StatusIs(absl::StatusCode::kResourceExhausted));

  size_t total = internal::kMaxProtobufSerializedBytes - 10;
  EXPECT_THAT(internal::AddWireBytesChecked(&total, 10, "test"), IsOk());
  EXPECT_EQ(total, internal::kMaxProtobufSerializedBytes);
  EXPECT_THAT(internal::AddWireBytesChecked(&total, 1, "test"),
              StatusIs(absl::StatusCode::kResourceExhausted));
  size_t zero_total = 0;
  EXPECT_THAT(
      internal::AddWireBytesChecked(
          &zero_total, internal::kMaxProtobufSerializedBytes + 1, "test"),
      StatusIs(absl::StatusCode::kResourceExhausted));
  // Caller-supplied max_bytes above kMaxProtobufSerializedBytes must clamp to
  // Protobuf's hard 2 GiB limit rather than relaxing it.
  EXPECT_THAT(internal::ValidateProtobufMessageSize(
                  internal::kMaxProtobufSerializedBytes + 1, "test",
                  internal::kMaxProtobufSerializedBytes + 100),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(internal::AddWireBytesChecked(
                  &zero_total, internal::kMaxProtobufSerializedBytes + 1,
                  "test", internal::kMaxProtobufSerializedBytes + 100),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(DeltaSeriesProtoConverterTest,
     SerializeAndCompressHeaderFirstParityAllFields) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("op_a");
  response.add_interned_strings("async_a");
  response.add_interned_strings("cnt_a");
  response.set_full_timespan_start_ps(100);
  response.set_full_timespan_end_ps(5000);
  xprof::DetailItem* detail = response.add_details();
  detail->set_name("flag_x");
  detail->set_value(true);
  xprof::Process* proc = response.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("Device0");
  xprof::TraceEventSeries* comp = response.add_complete_events();
  comp->mutable_metadata()->set_process_id(1);
  comp->mutable_metadata()->set_thread_id(2);
  comp->add_deltas(100);
  comp->add_durations(50);
  comp->add_name_refs(1);
  xprof::TraceEventSeries* async = response.add_async_events();
  async->mutable_metadata()->set_process_id(1);
  async->mutable_metadata()->set_name_ref(2);
  async->add_deltas(200);
  async->add_durations(30);
  xprof::TraceEventSeries* cnt = response.add_counter_events();
  cnt->mutable_metadata()->set_process_id(1);
  cnt->mutable_metadata()->set_name_ref(3);
  cnt->add_deltas(300);
  cnt->add_event_metadata()->set_counter_value_double(42.5);

  ASSERT_OK_AND_ASSIGN(const std::string from_const,
                       SerializeAndCompress(response));
  ASSERT_OK_AND_ASSIGN(const std::string from_header_first,
                       SerializeTraceDataResponseHeaderFirst(response));
  EXPECT_EQ(from_const, from_header_first);

  ASSERT_OK_AND_ASSIGN(const std::string decompressed,
                       ZstdCompression::Decompress(from_const));
  // Header-first wire invariant: first tag must not be Field 1 (0x0a).
  ASSERT_FALSE(decompressed.empty());
  EXPECT_NE(static_cast<uint8_t>(decompressed[0]), 0x0au);
  xprof::TraceDataResponse roundtrip;
  ASSERT_TRUE(roundtrip.ParseFromString(decompressed));
  EXPECT_THAT(roundtrip, EqualsProto(response));

  // Empty response (no metadata, total_wire_bytes == 0) round-trips
  // identically, and a non-empty response without metadata exercises
  // has_metadata() == false.
  xprof::TraceDataResponse empty_resp;
  ASSERT_OK_AND_ASSIGN(const std::string empty_const,
                       SerializeAndCompress(empty_resp));
  ASSERT_OK_AND_ASSIGN(const std::string empty_header_first,
                       SerializeTraceDataResponseHeaderFirst(empty_resp));
  EXPECT_EQ(empty_const, empty_header_first);
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse empty_rt,
                       DecompressAndParseResponse(empty_const));
  EXPECT_THAT(empty_rt, EqualsProto(empty_resp));

  xprof::TraceDataResponse no_meta_resp;
  no_meta_resp.set_full_timespan_start_ps(123);
  ASSERT_OK_AND_ASSIGN(const std::string no_meta_bytes,
                       SerializeTraceDataResponseHeaderFirst(no_meta_resp));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse no_meta_rt,
                       DecompressAndParseResponse(no_meta_bytes));
  EXPECT_THAT(no_meta_rt, EqualsProto(no_meta_resp));

  // Standalone StreamingDeltaSeriesProtoConverter framing & EOS coverage.
  EXPECT_THAT(StreamingDeltaSeriesProtoConverter::AppendCompressedFrame(
                  response, /*include_magic_header=*/true, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  StreamingDeltaSeriesProtoConverter::AppendEndOfStreamFrame(nullptr);
  std::string xchn_frames;
  ASSERT_THAT(StreamingDeltaSeriesProtoConverter::AppendCompressedFrame(
                  response, /*include_magic_header=*/true, &xchn_frames),
              IsOk());
  ASSERT_THAT(StreamingDeltaSeriesProtoConverter::AppendCompressedFrame(
                  no_meta_resp, /*include_magic_header=*/false, &xchn_frames),
              IsOk());
  StreamingDeltaSeriesProtoConverter::AppendEndOfStreamFrame(&xchn_frames);

  auto decode_le_u32 = [](absl::string_view bytes, size_t offset) -> uint32_t {
    return static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 1]))
            << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 2]))
            << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 3]))
            << 24);
  };

  ASSERT_GE(xchn_frames.size(), kXChunkMagic.size() + 4);
  EXPECT_EQ(absl::string_view(xchn_frames).substr(0, kXChunkMagic.size()),
            kXChunkMagic);

  // Frame 1: starts right after the 4-byte magic header.
  const size_t frame1_len_offset = kXChunkMagic.size();
  const uint32_t frame1_len = decode_le_u32(xchn_frames, frame1_len_offset);
  ASSERT_GT(frame1_len, 0u);
  const size_t frame1_payload_offset = frame1_len_offset + 4;
  const size_t frame1_end = frame1_payload_offset + frame1_len;
  ASSERT_LE(frame1_end + 4, xchn_frames.size());
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse frame1_rt,
                       DecompressAndParseResponse(
                           absl::string_view(xchn_frames)
                               .substr(frame1_payload_offset, frame1_len)));
  EXPECT_THAT(frame1_rt, EqualsProto(response));

  // Frame 2: immediately follows Frame 1 with no duplicate magic header.
  EXPECT_NE(
      absl::string_view(xchn_frames).substr(frame1_end, kXChunkMagic.size()),
      kXChunkMagic);
  const uint32_t frame2_len = decode_le_u32(xchn_frames, frame1_end);
  ASSERT_GT(frame2_len, 0u);
  const size_t frame2_payload_offset = frame1_end + 4;
  const size_t frame2_end = frame2_payload_offset + frame2_len;
  ASSERT_EQ(frame2_end + 4, xchn_frames.size());
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse frame2_rt,
                       DecompressAndParseResponse(
                           absl::string_view(xchn_frames)
                               .substr(frame2_payload_offset, frame2_len)));
  EXPECT_THAT(frame2_rt, EqualsProto(no_meta_resp));

  // End-of-stream sentinel: exact 4-byte little-endian 0 right after Frame 2.
  EXPECT_EQ(decode_le_u32(xchn_frames, frame2_end), 0u);
}

TEST(DeltaSeriesProtoConverterTest, InternalHelpersGroupIdRunsAndFlowCategory) {
  // IsDataMotionLayerThreadName: all 4 canonical constants + non-match.
  EXPECT_TRUE(internal::IsDataMotionLayerThreadName(
      internal::kComputeUtilizationRooflineThreadName));
  EXPECT_TRUE(internal::IsDataMotionLayerThreadName(
      internal::kComputeUtilizationThreadName));
  EXPECT_TRUE(internal::IsDataMotionLayerThreadName(
      internal::kDataMotionLayersLowerThreadName));
  EXPECT_TRUE(internal::IsDataMotionLayerThreadName(
      internal::kDataMotionLayersTitleThreadName));
  EXPECT_FALSE(internal::IsDataMotionLayerThreadName("XLA Ops"));

  // ClassifyHloExtractionTrack: XLA Ops, all 4 data-motion constants, regular,
  // and empty resource name.
  EXPECT_EQ(internal::ClassifyHloExtractionTrack("XLA Ops"),
            internal::HloExtractionTrackKind::kXlaOps);
  EXPECT_EQ(internal::ClassifyHloExtractionTrack(
                internal::kComputeUtilizationRooflineThreadName),
            internal::HloExtractionTrackKind::kDataMotionLayer);
  EXPECT_EQ(internal::ClassifyHloExtractionTrack(
                internal::kComputeUtilizationThreadName),
            internal::HloExtractionTrackKind::kDataMotionLayer);
  EXPECT_EQ(internal::ClassifyHloExtractionTrack(
                internal::kDataMotionLayersLowerThreadName),
            internal::HloExtractionTrackKind::kDataMotionLayer);
  EXPECT_EQ(internal::ClassifyHloExtractionTrack(
                internal::kDataMotionLayersTitleThreadName),
            internal::HloExtractionTrackKind::kDataMotionLayer);
  EXPECT_EQ(internal::ClassifyHloExtractionTrack("Stream 1"),
            internal::HloExtractionTrackKind::kRegular);
  EXPECT_EQ(internal::ClassifyHloExtractionTrack(""),
            internal::HloExtractionTrackKind::kRegular);

  // TraceNameRefInternCache: miss, 1-slot MRU hit, map hit after eviction,
  // uncached missing ref, inline name bypass, per-trace binding, and Clear().
  {
    Trace trace_a;
    (*trace_a.mutable_name_table())[1] = "op_alpha";
    (*trace_a.mutable_name_table())[2] = "op_beta";
    DeltaSeriesStringTable string_table;
    int intern_calls = 0;
    auto intern_fn = [&](absl::string_view s) {
      ++intern_calls;
      return string_table.InternString(s);
    };

    internal::TraceNameRefInternCache cache(&trace_a);
    TraceEvent ev1;
    ev1.set_name_ref(1);
    TraceEvent ev2;
    ev2.set_name_ref(2);

    // Cold miss for ref 1.
    const internal::TraceNameRefInternCache::CachedEntry e1_miss =
        cache.ResolveAndIntern(ev1, intern_fn);
    EXPECT_EQ(e1_miss.resolved_name, "op_alpha");
    EXPECT_GT(e1_miss.interned_id, 0u);
    EXPECT_EQ(intern_calls, 1);

    // 1-slot MRU hit for ref 1 does not invoke intern_fn.
    const internal::TraceNameRefInternCache::CachedEntry e1_mru =
        cache.ResolveAndIntern(ev1, intern_fn);
    EXPECT_EQ(e1_mru.interned_id, e1_miss.interned_id);
    EXPECT_EQ(e1_mru.resolved_name, "op_alpha");
    EXPECT_EQ(intern_calls, 1);

    // Cold miss for ref 2 evicts ref 1 from MRU slot.
    const internal::TraceNameRefInternCache::CachedEntry e2_miss =
        cache.ResolveAndIntern(ev2, intern_fn);
    EXPECT_EQ(e2_miss.resolved_name, "op_beta");
    EXPECT_NE(e2_miss.interned_id, e1_miss.interned_id);
    EXPECT_EQ(intern_calls, 2);

    // Map hit for ref 1 after MRU eviction still avoids intern_fn.
    const internal::TraceNameRefInternCache::CachedEntry e1_map =
        cache.ResolveAndIntern(ev1, intern_fn);
    EXPECT_EQ(e1_map.interned_id, e1_miss.interned_id);
    EXPECT_EQ(e1_map.resolved_name, "op_alpha");
    EXPECT_EQ(intern_calls, 2);

    // Missing name_ref returns "" and is NOT cached in map_.
    TraceEvent ev_missing;
    ev_missing.set_name_ref(999);
    const internal::TraceNameRefInternCache::CachedEntry m1 =
        cache.ResolveAndIntern(ev_missing, intern_fn);
    EXPECT_EQ(m1.resolved_name, "");
    EXPECT_EQ(m1.interned_id, 0u);
    EXPECT_EQ(intern_calls, 3);
    const internal::TraceNameRefInternCache::CachedEntry m2 =
        cache.ResolveAndIntern(ev_missing, intern_fn);
    EXPECT_EQ(m2.resolved_name, "");
    EXPECT_EQ(m2.interned_id, 0u);
    EXPECT_EQ(intern_calls, 4);

    // Inline name bypasses the cache every time.
    TraceEvent ev_inline;
    ev_inline.set_name("inline_op");
    const internal::TraceNameRefInternCache::CachedEntry inl1 =
        cache.ResolveAndIntern(ev_inline, intern_fn);
    EXPECT_EQ(inl1.resolved_name, "inline_op");
    EXPECT_EQ(intern_calls, 5);
    const internal::TraceNameRefInternCache::CachedEntry inl2 =
        cache.ResolveAndIntern(ev_inline, intern_fn);
    EXPECT_EQ(inl2.interned_id, inl1.interned_id);
    EXPECT_EQ(intern_calls, 6);

    // Clear() resets both MRU slot and map_.
    cache.Clear();
    const internal::TraceNameRefInternCache::CachedEntry e1_after_clear =
        cache.ResolveAndIntern(ev1, intern_fn);
    EXPECT_EQ(e1_after_clear.interned_id, e1_miss.interned_id);
    EXPECT_EQ(intern_calls, 7);
    const internal::TraceNameRefInternCache::CachedEntry e2_after_clear =
        cache.ResolveAndIntern(ev2, intern_fn);
    EXPECT_EQ(e2_after_clear.interned_id, e2_miss.interned_id);
    EXPECT_EQ(intern_calls, 8);

    // Separate Trace bound to a separate cache resolves colliding ref 1
    // independently.
    Trace trace_b;
    (*trace_b.mutable_name_table())[1] = "op_gamma";
    internal::TraceNameRefInternCache cache_b(&trace_b);
    const internal::TraceNameRefInternCache::CachedEntry b1 =
        cache_b.ResolveAndIntern(ev1, intern_fn);
    EXPECT_EQ(b1.resolved_name, "op_gamma");
    EXPECT_NE(b1.interned_id, e1_miss.interned_id);
  }

  // ResolveResourceName.
  Trace trace;
  Device dev;
  (*dev.mutable_resources())[1].set_name("Stream 1");
  (*dev.mutable_resources())[2];  // Resource without name.
  (*trace.mutable_devices())[0] = dev;
  EXPECT_EQ(internal::ResolveResourceName(trace, /*pid=*/99, /*tid=*/1), "");
  EXPECT_EQ(internal::ResolveResourceName(trace, /*pid=*/0, /*tid=*/99), "");
  EXPECT_EQ(internal::ResolveResourceName(trace, /*pid=*/0, /*tid=*/2), "");
  EXPECT_EQ(internal::ResolveResourceName(trace, /*pid=*/0, /*tid=*/1),
            "Stream 1");

  // GroupIdRunBuilder: empty flush emits nothing; repeated values merge;
  // transitions flush prior run; second flush is idempotent.
  {
    xprof::TraceEventSeries empty_series;
    internal::GroupIdRunBuilder empty_builder;
    empty_builder.Flush(&empty_series);
    EXPECT_THAT(empty_series.group_id_run_values(), IsEmpty());
    EXPECT_THAT(empty_series.group_id_run_lengths(), IsEmpty());
  }
  {
    xprof::TraceEventSeries series;
    internal::GroupIdRunBuilder builder;
    builder.Add(-1, &series);
    builder.Add(-1, &series);
    builder.Add(42, &series);
    builder.Add(1LL << 40, &series);
    builder.Add(1LL << 40, &series);
    builder.Add(1LL << 40, &series);
    builder.Flush(&series);
    builder.Flush(&series);  // Idempotent no-op after flush.
    EXPECT_THAT(series.group_id_run_values(), ElementsAre(-1, 42, 1LL << 40));
    EXPECT_THAT(series.group_id_run_lengths(), ElementsAre(2, 1, 3));
  }

  // HasNonGenericFlowCategory: unset, kGeneric, kLegacy, out-of-range, and
  // valid non-generic categories.
  {
    TraceEvent ev;
    EXPECT_FALSE(internal::HasNonGenericFlowCategory(ev));

    ev.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kGeneric));
    EXPECT_FALSE(internal::HasNonGenericFlowCategory(ev));

    ev.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kLegacy));
    EXPECT_FALSE(internal::HasNonGenericFlowCategory(ev));

    // Out-of-range category maps to kGeneric via GetSafeContextType.
    ev.set_flow_category(256u + 2u);
    EXPECT_FALSE(internal::HasNonGenericFlowCategory(ev));

    ev.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kTfExecutor));
    EXPECT_TRUE(internal::HasNonGenericFlowCategory(ev));
  }
}

TEST(DeltaSeriesProtoConverterTest,
     ExtractHloStringsAndDirectConverterEdgeCases) {
  Trace trace;
  (*trace.mutable_name_table())[1] = "valid_op";
  DeltaSeriesProtoConverter::RawDataArgsExtractor parse_raw_data =
      internal::MakeDefaultRawDataArgsExtractor<RawData>();
  std::string out_op;
  std::string out_mod;

  // Missing/empty raw_data, XLA Ops wire pre-filter, corrupt raw_data,
  // proto without args, and XLA Ops single-key HLO extraction.
  {
    TraceEvent ev_no_raw;
    EXPECT_FALSE(internal::ExtractHloStringsFromRawData(
        trace, ev_no_raw, internal::HloExtractionTrackKind::kXlaOps, "ev",
        parse_raw_data, &out_op, &out_mod));
    TraceEvent ev;
    ev.set_raw_data("");
    EXPECT_FALSE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kXlaOps, "ev",
        parse_raw_data, &out_op, &out_mod));
    // Unrelated args short-circuit on the XLA Ops wire pre-filter.
    ev.set_raw_data(MakeRawDataWithStrArgs({{"long_name", "not_hlo"}}));
    EXPECT_FALSE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kXlaOps, "ev",
        parse_raw_data, &out_op, &out_mod));
    // Corrupt wire bytes containing `hlo_op` pass the pre-filter and hit
    // ParseFromString failure.
    ev.set_raw_data("hlo_op_\xff\xff\xff");
    EXPECT_FALSE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kXlaOps, "ev",
        parse_raw_data, &out_op, &out_mod));
    // Valid RawData without args (dma_activity instead of args).
    RawData dma_only;
    dma_only.mutable_dma_activity()->set_kilobytes(4);
    ev.set_raw_data(dma_only.SerializeAsString());
    EXPECT_FALSE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kDataMotionLayer, "ev",
        parse_raw_data, &out_op, &out_mod));
    // XLA Ops with hlo_op only (distinct from event_name) and hlo_module only.
    ev.set_raw_data(MakeRawDataWithStrArgs({{"hlo_op", "only_op"}}));
    EXPECT_TRUE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kXlaOps, "ev",
        parse_raw_data, &out_op, &out_mod));
    EXPECT_EQ(out_op, "only_op");
    EXPECT_EQ(out_mod, "");

    ev.set_raw_data(MakeRawDataWithStrArgs({{"hlo_module", "only_mod"}}));
    EXPECT_TRUE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kXlaOps, "ev",
        parse_raw_data, &out_op, &out_mod));
    EXPECT_EQ(out_op, "");
    EXPECT_EQ(out_mod, "only_mod");
  }

  // Regular track containing "hlo_op" and "hlo_module" substrings inside an
  // unrelated arg value passes the fast substring check, parses, then exits on
  // !has_hlo_in_args.
  {
    TraceEvent ev;
    ev.set_raw_data(
        MakeRawDataWithStrArgs({{"note", "hlo_op and hlo_module"}}));
    EXPECT_FALSE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kRegular, "ev",
        parse_raw_data, &out_op, &out_mod));
  }

  // Same-length decoy arg names (4, 6, 10, 13, 14), dangling ref_value,
  // uint_value program_id, and hlo_module suffix mismatch cases.
  {
    TraceEvent ev;
    RawData rd;
    auto add_str = [&](absl::string_view k, absl::string_view v) {
      TraceEventArguments::Argument* a = rd.mutable_args()->add_arg();
      a->set_name(std::string(k));
      a->set_str_value(std::string(v));
    };
    add_str("abcd", "x");
    add_str("abcdef", "x");
    add_str("0123456789", "x");
    add_str("13_char_dummy", "x");
    add_str("14_char_dummy!", "x");
    TraceEventArguments::Argument* dbl_arg = rd.mutable_args()->add_arg();
    dbl_arg->set_name("hlo_op");
    dbl_arg->set_double_value(1.5);
    TraceEventArguments::Argument* bad_ref = rd.mutable_args()->add_arg();
    bad_ref->set_name("hlo_op");
    bad_ref->set_ref_value(999);
    TraceEventArguments::Argument* good_ref = rd.mutable_args()->add_arg();
    good_ref->set_name("hlo_op");
    good_ref->set_ref_value(1);
    add_str("hlo_module", "a");  // Shorter than suffix "(1234)".
    TraceEventArguments::Argument* pid = rd.mutable_args()->add_arg();
    pid->set_name("program_id");
    pid->set_uint_value(1234);
    ev.set_raw_data(rd.SerializeAsString());
    EXPECT_TRUE(internal::ExtractHloStringsFromRawData(
        trace, ev, internal::HloExtractionTrackKind::kDataMotionLayer, "ev",
        parse_raw_data, &out_op, &out_mod));
    EXPECT_EQ(out_op, "valid_op");
    EXPECT_EQ(out_mod, "a(1234)");
  }
  {
    // hlo_module with missing ')' and with different id inside parens.
    TraceEvent ev1;
    ev1.set_raw_data(MakeRawDataWithStrArgs(
        {{"hlo_op", "op"}, {"hlo_module", "mod(88"}, {"hlo_module_id", "88"}}));
    EXPECT_TRUE(internal::ExtractHloStringsFromRawData(
        trace, ev1, internal::HloExtractionTrackKind::kRegular, "ev",
        parse_raw_data, &out_op, &out_mod));
    EXPECT_EQ(out_mod, "mod(88(88)");

    TraceEvent ev2;
    ev2.set_raw_data(MakeRawDataWithStrArgs({{"hlo_op", "op"},
                                             {"hlo_module", "mod(77)"},
                                             {"hlo_module_id", "88"}}));
    EXPECT_TRUE(internal::ExtractHloStringsFromRawData(
        trace, ev2, internal::HloExtractionTrackKind::kRegular, "ev",
        parse_raw_data, &out_op, &out_mod));
    EXPECT_EQ(out_mod, "mod(77)(88)");

    TraceEvent ev3;
    ev3.set_raw_data(MakeRawDataWithStrArgs({{"hlo_op", "op"},
                                             {"hlo_module", "mod(88X"},
                                             {"hlo_module_id", "88"}}));
    EXPECT_TRUE(internal::ExtractHloStringsFromRawData(
        trace, ev3, internal::HloExtractionTrackKind::kRegular, "ev",
        parse_raw_data, &out_op, &out_mod));
    EXPECT_EQ(out_mod, "mod(88X(88)");
  }

  // Sparse complete + async converter edge cases and empty-track skipping in
  // GenerateResponse.
  {
    Trace t;
    Device dev;
    dev.set_name("TPU:0");
    (*dev.mutable_resources())[1].set_name("Main");
    (*t.mutable_devices())[0] = dev;

    ContainerWithEmptyTrack container(t);
    TraceEvent comp_generic;
    comp_generic.set_name("cg");
    comp_generic.set_timestamp_ps(100);
    comp_generic.set_duration_ps(10);
    comp_generic.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kGeneric));
    container.AddCompleteEvent(0, 1, &comp_generic);

    TraceEvent comp_legacy;
    comp_legacy.set_name("cl");
    comp_legacy.set_timestamp_ps(200);
    comp_legacy.set_duration_ps(10);
    comp_legacy.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kLegacy));
    container.AddCompleteEvent(0, 1, &comp_legacy);

    // Async sparse events covering: kGeneric, kLegacy, kTfExecutor, dur>0 with
    // FLOW_NONE vs FLOW_START, dur==0 with FLOW_NONE, serial!=0, and HLO
    // raw_data.
    constexpr uint64_t kWideFlowId = (uint64_t{1} << 32) + 11;

    TraceEvent a0;
    a0.set_timestamp_ps(100);
    a0.set_duration_ps(50);
    a0.set_flow_id(10);
    a0.set_flow_entry_type(TraceEvent::FLOW_NONE);
    a0.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kGeneric));
    container.AddAsyncEvent(0, "AsyncEdge", &a0);

    TraceEvent a1;
    a1.set_timestamp_ps(200);
    a1.set_duration_ps(50);
    a1.set_flow_id(kWideFlowId);
    a1.set_flow_entry_type(TraceEvent::FLOW_START);
    a1.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kLegacy));
    container.AddAsyncEvent(0, "AsyncEdge", &a1);

    TraceEvent a2;
    a2.set_timestamp_ps(300);
    a2.set_duration_ps(50);
    a2.set_flow_id(12);
    a2.set_flow_entry_type(TraceEvent::FLOW_NONE);
    a2.set_serial(77);
    container.AddAsyncEvent(0, "AsyncEdge", &a2);

    TraceEvent a3;
    a3.set_timestamp_ps(400);
    a3.set_duration_ps(50);
    a3.set_flow_id(13);
    a3.set_flow_entry_type(TraceEvent::FLOW_NONE);
    a3.set_flow_category(
        static_cast<uint32_t>(tsl::profiler::ContextType::kTfExecutor));
    container.AddAsyncEvent(0, "AsyncEdge", &a3);

    TraceEvent a4;
    a4.set_timestamp_ps(500);
    a4.set_duration_ps(50);
    a4.set_flow_id(14);
    a4.set_flow_entry_type(TraceEvent::FLOW_NONE);
    a4.set_raw_data(MakeRawDataWithStrArgs(
        {{"hlo_op", "async_op"}, {"hlo_module", "async_mod"}}));
    container.AddAsyncEvent(0, "AsyncEdge", &a4);

    TraceEvent a5;
    a5.set_timestamp_ps(600);
    a5.set_duration_ps(50);
    container.AddAsyncEvent(0, "AsyncEdge", &a5);

    TraceEvent a6;
    a6.set_timestamp_ps(700);
    a6.set_duration_ps(0);
    a6.set_flow_id(16);
    a6.set_flow_entry_type(TraceEvent::FLOW_NONE);
    container.AddAsyncEvent(0, "AsyncEdge", &a6);

    DeltaSeriesProtoConversionOptions opts;
    opts.sparse_event_metadata = true;
    ASSERT_OK_AND_ASSIGN(
        const std::string mono,
        ConvertTraceDataToCompressedDeltaSeriesProto(opts, container));
    ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse resp,
                         DecompressAndParseResponse(mono));
    ASSERT_EQ(resp.complete_events_size(), 1);
    EXPECT_EQ(resp.complete_events(0).event_metadata_size(), 0);
    ASSERT_EQ(resp.async_events_size(), 1);
    const xprof::TraceEventSeries& async_series = resp.async_events(0);
    EXPECT_THAT(async_series.event_metadata_indices(),
                ElementsAre(1, 2, 3, 4, 6));
    ASSERT_EQ(async_series.event_metadata_size(), 5);
    EXPECT_EQ(async_series.event_metadata(0).flow_id(), kWideFlowId);
    EXPECT_EQ(async_series.event_metadata(1).flow_id(), 0u);
    EXPECT_EQ(async_series.event_metadata(1).serial(), 77u);
    EXPECT_EQ(async_series.event_metadata(2).flow_id(), 0u);
    EXPECT_EQ(
        resp.interned_strings(async_series.event_metadata(2).flow_category()),
        "tf_exec");
    EXPECT_EQ(async_series.event_metadata(3).flow_id(), 0u);
    EXPECT_EQ(
        resp.interned_strings(async_series.event_metadata(3).hlo_op_ref()),
        "async_op");
    EXPECT_EQ(
        resp.interned_strings(async_series.event_metadata(3).hlo_module_ref()),
        "async_mod");
    EXPECT_EQ(async_series.event_metadata(4).flow_id(), 16u);

    // Dense mode retains original flow_ids on all 7 async events.
    DeltaSeriesProtoConversionOptions dense_opts;
    dense_opts.sparse_event_metadata = false;
    ASSERT_OK_AND_ASSIGN(
        const std::string dense_mono,
        ConvertTraceDataToCompressedDeltaSeriesProto(dense_opts, container));
    ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse dense_resp,
                         DecompressAndParseResponse(dense_mono));
    ASSERT_EQ(dense_resp.async_events_size(), 1);
    const xprof::TraceEventSeries& dense_async = dense_resp.async_events(0);
    ASSERT_EQ(dense_async.event_metadata_size(), 7);
    EXPECT_EQ(dense_async.event_metadata(0).flow_id(), 10u);
    EXPECT_EQ(dense_async.event_metadata(1).flow_id(), kWideFlowId);
    EXPECT_EQ(dense_async.event_metadata(2).flow_id(), 12u);
    EXPECT_EQ(dense_async.event_metadata(3).flow_id(), 13u);
    EXPECT_EQ(dense_async.event_metadata(4).flow_id(), 14u);
    EXPECT_EQ(dense_async.event_metadata(5).flow_id(), 0u);
    EXPECT_EQ(dense_async.event_metadata(6).flow_id(), 16u);
  }
}

TEST(DeltaSeriesProtoConverterTest,
     XlaOpsTrackOver64EventsExtractsLaterHloMetadataAndPreservesOrder) {
  auto build_and_verify = [](size_t num_events) {
    SCOPED_TRACE(absl::StrCat("num_events=", num_events));
    Trace trace;
    Device dev;
    dev.set_name("/device:TPU:0");
    (*dev.mutable_resources())[1].set_name("XLA Ops");
    (*dev.mutable_resources())[2].set_name("XLA Ops");
    (*trace.mutable_devices())[0] = dev;

    TestTraceEventsContainer container(trace);

    // Track 1: event 0 has unrelated raw_data, event 1 has hlo_op ==
    // event_name without hlo_module, and events 2 and last carry real HLO
    // metadata.
    const size_t last_idx = num_events - 1;
    std::vector<TraceEvent> track1(num_events);
    for (size_t i = 0; i < num_events; ++i) {
      track1[i].set_timestamp_ps(1000 + i * 10);
      track1[i].set_duration_ps(5);
      if (i == 0) {
        track1[i].set_name("ev0_unrelated");
        track1[i].set_raw_data(
            MakeRawDataWithStrArgs({{"long_name", "unrelated_only"}}));
      } else if (i == 1) {
        track1[i].set_name("ev1_same_op");
        track1[i].set_raw_data(
            MakeRawDataWithStrArgs({{"hlo_op", "ev1_same_op"}}));
      } else if (i == 2) {
        track1[i].set_name("ev2_wrapper");
        track1[i].set_raw_data(MakeRawDataWithStrArgs(
            {{"hlo_op", "ev2_hlo_op"}, {"hlo_module", "ev2_hlo_mod"}}));
      } else if (i == last_idx) {
        track1[i].set_name("ev_last_wrapper");
        track1[i].set_raw_data(MakeRawDataWithStrArgs(
            {{"hlo_op", "ev_last_hlo_op"}, {"hlo_module", "ev_last_hlo_mod"}}));
      } else {
        track1[i].set_name("ev_mid");
      }
      container.AddCompleteEvent(0, 1, &track1[i]);
    }

    // Track 2: event 0 has no raw_data at all, while event (last_idx - 1)
    // carries HLO metadata.
    const size_t t2_hlo_idx = num_events - 2;
    std::vector<TraceEvent> track2(num_events);
    for (size_t i = 0; i < num_events; ++i) {
      track2[i].set_timestamp_ps(5000 + i * 10);
      track2[i].set_duration_ps(5);
      if (i == t2_hlo_idx) {
        track2[i].set_name("t2_late_wrapper");
        track2[i].set_raw_data(MakeRawDataWithStrArgs(
            {{"hlo_op", "t2_hlo_op"}, {"hlo_module", "t2_hlo_mod"}}));
      } else {
        track2[i].set_name("t2_no_raw");
      }
      container.AddCompleteEvent(0, 2, &track2[i]);
    }

    DeltaSeriesProtoConversionOptions opts_1t;
    opts_1t.sparse_event_metadata = true;

    ASSERT_OK_AND_ASSIGN(
        const std::string mono_compressed,
        ConvertTraceDataToCompressedDeltaSeriesProto(opts_1t, container));
    ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse mono_resp,
                         DecompressAndParseResponse(mono_compressed));

    // Verify canonical event-order string interning in GenerateResponse
    // (earlier event names precede later event HLO strings).
    auto verify_intern_order =
        [](const absl::flat_hash_map<absl::string_view, int>& str_idx) {
          auto idx_of = [&](absl::string_view key) -> int {
            const absl::flat_hash_map<absl::string_view, int>::const_iterator
                it = str_idx.find(key);
            return it != str_idx.end() ? it->second : -1;
          };
          const int idx_ev0 = idx_of("ev0_unrelated");
          const int idx_ev1 = idx_of("ev1_same_op");
          const int idx_ev2 = idx_of("ev2_wrapper");
          const int idx_ev2_op = idx_of("ev2_hlo_op");
          const int idx_ev2_mod = idx_of("ev2_hlo_mod");
          const int idx_last = idx_of("ev_last_wrapper");
          const int idx_last_op = idx_of("ev_last_hlo_op");
          const int idx_last_mod = idx_of("ev_last_hlo_mod");
          const int idx_t2_no_raw = idx_of("t2_no_raw");
          const int idx_t2_late = idx_of("t2_late_wrapper");
          const int idx_t2_op = idx_of("t2_hlo_op");
          const int idx_t2_mod = idx_of("t2_hlo_mod");
          EXPECT_GT(idx_ev0, 0);
          EXPECT_LT(idx_ev0, idx_ev1);
          EXPECT_LT(idx_ev1, idx_ev2);
          EXPECT_LT(idx_ev2, idx_ev2_op);
          EXPECT_LT(idx_ev2_op, idx_ev2_mod);
          EXPECT_LT(idx_ev2_mod, idx_last);
          EXPECT_LT(idx_last, idx_last_op);
          EXPECT_LT(idx_last_op, idx_last_mod);
          EXPECT_LT(idx_last_mod, idx_t2_no_raw);
          EXPECT_LT(idx_t2_no_raw, idx_t2_late);
          EXPECT_LT(idx_t2_late, idx_t2_op);
          EXPECT_LT(idx_t2_op, idx_t2_mod);
        };
    absl::flat_hash_map<absl::string_view, int> mono_str_idx;
    for (int idx = 0; idx < mono_resp.interned_strings_size(); ++idx) {
      mono_str_idx.try_emplace(mono_resp.interned_strings(idx), idx);
    }
    verify_intern_order(mono_str_idx);

    // Verify GenerateResponse extracted exact HLO metadata on both tracks.
    ASSERT_EQ(mono_resp.complete_events_size(), 2);
    absl::flat_hash_map<std::pair<uint64_t, uint32_t>,
                        std::pair<std::string, std::string>>
        mono_hlo;
    for (const xprof::TraceEventSeries& series : mono_resp.complete_events()) {
      const uint64_t tid = series.metadata().thread_id();
      ASSERT_EQ(series.event_metadata_indices_size(),
                series.event_metadata_size());
      for (int m = 0; m < series.event_metadata_size(); ++m) {
        const xprof::TraceEventMetadata& meta = series.event_metadata(m);
        const uint32_t idx =
            static_cast<uint32_t>(series.event_metadata_indices(m));
        const std::pair<std::string, std::string> hlo_strings(
            mono_resp.interned_strings(meta.hlo_op_ref()),
            mono_resp.interned_strings(meta.hlo_module_ref()));
        EXPECT_TRUE(mono_hlo.try_emplace({tid, idx}, hlo_strings).second);
      }
    }
    EXPECT_THAT(
        mono_hlo,
        UnorderedElementsAre(
            Pair(Pair(uint64_t{1}, uint32_t{2}),
                 Pair("ev2_hlo_op", "ev2_hlo_mod")),
            Pair(Pair(uint64_t{1}, static_cast<uint32_t>(last_idx)),
                 Pair("ev_last_hlo_op", "ev_last_hlo_mod")),
            Pair(Pair(uint64_t{2}, static_cast<uint32_t>(t2_hlo_idx)),
                 Pair("t2_hlo_op", "t2_hlo_mod"))));
  };

  // Both >64-event and <=64-event XLA Ops tracks must behave identically.
  build_and_verify(/*num_events=*/70);
  build_and_verify(/*num_events=*/10);
}

TEST(DeltaSeriesProtoConverterTest,
     SerializeTraceDataResponseHeaderFirstWithLimitEnforcesWireBounds) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("op_name");
  response.add_interned_strings("async_name");
  response.add_interned_strings("counter_name");
  response.set_full_timespan_start_ps(12345);
  response.set_full_timespan_end_ps(67890);

  xprof::DetailItem* detail = response.add_details();
  detail->set_name("detail_key");
  detail->set_value(true);

  xprof::Process* proc = response.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("TPU:0");

  xprof::TraceEventSeries* cnt = response.add_counter_events();
  cnt->mutable_metadata()->set_process_id(1);
  cnt->mutable_metadata()->set_name_ref(3);
  cnt->add_deltas(100);
  cnt->add_event_metadata()->set_counter_value_double(7.5);

  xprof::TraceEventSeries* async = response.add_async_events();
  async->mutable_metadata()->set_process_id(1);
  async->mutable_metadata()->set_name_ref(2);
  async->add_deltas(200);
  async->add_durations(50);

  xprof::TraceEventSeries* comp = response.add_complete_events();
  comp->mutable_metadata()->set_process_id(1);
  comp->mutable_metadata()->set_thread_id(1);
  comp->add_deltas(300);
  comp->add_durations(25);
  comp->add_name_refs(1);

  auto delimited_wire_size = [](size_t msg_size) -> size_t {
    return 1 + google::protobuf::io::CodedOutputStream::VarintSize64(msg_size) + msg_size;
  };

  const size_t cnt_msg_size = cnt->ByteSizeLong();
  const size_t after_cnt = delimited_wire_size(cnt_msg_size);
  const size_t after_async =
      after_cnt + delimited_wire_size(async->ByteSizeLong());
  const size_t after_meta =
      after_async + delimited_wire_size(response.metadata().ByteSizeLong());
  size_t after_strings = after_meta;
  for (const std::string& str : response.interned_strings()) {
    after_strings += delimited_wire_size(str.size());
  }
  const size_t after_details =
      after_strings + delimited_wire_size(detail->ByteSizeLong());
  const size_t after_start = after_details + 1 +
                             google::protobuf::io::CodedOutputStream::VarintSize64(
                                 response.full_timespan_start_ps());
  const size_t after_end = after_start + 1 +
                           google::protobuf::io::CodedOutputStream::VarintSize64(
                               response.full_timespan_end_ps());
  const size_t total_wire_bytes =
      after_end + delimited_wire_size(comp->ByteSizeLong());

  // 1. counter_events submessage exceeds ValidateProtobufMessageSize.
  EXPECT_THAT(internal::SerializeTraceDataResponseHeaderFirstWithLimit(
                  response, /*zstd_ctx=*/nullptr, cnt_msg_size - 1),
              StatusIs(absl::StatusCode::kResourceExhausted));
  // 2. counter_events fits submessage check but overflows delimited total.
  EXPECT_THAT(internal::SerializeTraceDataResponseHeaderFirstWithLimit(
                  response, /*zstd_ctx=*/nullptr, cnt_msg_size),
              StatusIs(absl::StatusCode::kResourceExhausted));
  // 3. Subsequent fields exceed cumulative total one by one.
  for (const size_t failing_limit :
       {after_async - 1, after_meta - 1, after_strings - 1, after_details - 1,
        after_start - 1, after_end - 1, total_wire_bytes - 1}) {
    EXPECT_THAT(internal::SerializeTraceDataResponseHeaderFirstWithLimit(
                    response, /*zstd_ctx=*/nullptr, failing_limit),
                StatusIs(absl::StatusCode::kResourceExhausted))
        << "failing_limit=" << failing_limit;
  }

  // Single interned string exceeding ValidateProtobufMessageSize directly.
  {
    xprof::TraceDataResponse str_only;
    str_only.add_interned_strings("long_string_value");
    EXPECT_THAT(internal::SerializeTraceDataResponseHeaderFirstWithLimit(
                    str_only, /*zstd_ctx=*/nullptr, /*max_wire_bytes=*/4),
                StatusIs(absl::StatusCode::kResourceExhausted));
  }

  // Exact boundary succeeds and round-trips identically.
  ASSERT_OK_AND_ASSIGN(
      const std::string exact_compressed,
      internal::SerializeTraceDataResponseHeaderFirstWithLimit(
          response, /*zstd_ctx=*/nullptr, total_wire_bytes));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse roundtrip,
                       DecompressAndParseResponse(exact_compressed));
  EXPECT_THAT(roundtrip, EqualsProto(response));

  // Moved-from ZstdCompressorContext propagates compression error.
  ZstdCompressorContext ctx(/*compression_level=*/1);
  ZstdCompressorContext moved_ctx = std::move(ctx);
  (void)moved_ctx;
  EXPECT_THAT(internal::SerializeTraceDataResponseHeaderFirstWithLimit(
                  response, &ctx, total_wire_bytes),
              StatusIs(absl::StatusCode::kInternal));
}

// Custom RawDataType encoding TraceEventArguments under protobuf field number 5
// (instead of OSS RawData's field 2) to verify container-generic HLO/counter
// extraction.
class TagFiveRawData {
 public:
  static constexpr uint32_t kArgsFieldTag = (5u << 3) | 2u;

  bool ParseFromString(absl::string_view raw_bytes) {
    has_args_ = false;
    args_.Clear();
    google::protobuf::io::CodedInputStream coded(
        reinterpret_cast<const uint8_t*>(raw_bytes.data()),
        static_cast<int>(raw_bytes.size()));
    while (const uint32_t tag = coded.ReadTag()) {
      if (tag == kArgsFieldTag) {
        uint32_t len = 0;
        if (!coded.ReadVarint32(&len)) return false;
        std::string payload;
        if (!coded.ReadString(&payload, len)) return false;
        if (!args_.ParseFromString(payload)) return false;
        has_args_ = true;
      } else {
        const uint32_t wire_type = tag & 7u;
        if (wire_type == 0u) {
          uint64_t val = 0;
          if (!coded.ReadVarint64(&val)) return false;
        } else if (wire_type == 1u) {
          if (!coded.Skip(8)) return false;
        } else if (wire_type == 2u) {
          uint32_t len = 0;
          if (!coded.ReadVarint32(&len) || !coded.Skip(len)) return false;
        } else if (wire_type == 5u) {
          if (!coded.Skip(4)) return false;
        } else {
          return false;
        }
      }
    }
    return coded.ConsumedEntireMessage();
  }

  bool has_args() const { return has_args_; }
  const TraceEventArguments& args() const { return args_; }
  TraceEventArguments* mutable_args() {
    has_args_ = true;
    return &args_;
  }

  std::string SerializeAsString() const {
    if (!has_args_) return "";
    const std::string payload = args_.SerializeAsString();
    const size_t varint_size =
        google::protobuf::io::CodedOutputStream::VarintSize64(payload.size());
    std::string out(1 + varint_size + payload.size(), '\0');
    uint8_t* ptr = reinterpret_cast<uint8_t*>(out.data());
    *ptr++ = static_cast<uint8_t>(kArgsFieldTag);
    ptr = google::protobuf::io::CodedOutputStream::WriteVarint64ToArray(payload.size(),
                                                              ptr);
    if (!payload.empty()) {
      std::copy(payload.begin(), payload.end(), reinterpret_cast<char*>(ptr));
    }
    return out;
  }

 private:
  bool has_args_ = false;
  TraceEventArguments args_;
};

class TestTagFiveTraceEventsContainer
    : public TraceEventsContainerBase<EventFactory, TagFiveRawData> {
 public:
  explicit TestTagFiveTraceEventsContainer(const Trace& trace) {
    trace_ = trace;
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

  void AddCounterEvent(uint32_t device_id, absl::string_view name,
                       TraceEvent* event) {
    event->set_device_id(device_id);
    event->set_name(name.data(), name.size());
    AddArenaEvent(event);
  }

  const Trace& trace() const { return trace_; }
};

std::string MakeTagFiveRawDataWithStrArgs(
    std::initializer_list<std::pair<absl::string_view, absl::string_view>>
        args) {
  TagFiveRawData rd;
  for (const auto& [name, value] : args) {
    TraceEventArguments::Argument* arg = rd.mutable_args()->add_arg();
    arg->set_name(std::string(name));
    arg->set_str_value(std::string(value));
  }
  return rd.SerializeAsString();
}

TEST(DeltaSeriesProtoConverterTest,
     CustomNonFieldTwoRawDataExtractsHloAndCounters) {
  // Confirm default field-2 RawData does not see field-5 args.
  {
    const std::string encoded = MakeTagFiveRawDataWithStrArgs(
        {{"hlo_op", "op_x"}, {"hlo_module", "mod_x"}});
    RawData field_two_rd;
    ASSERT_TRUE(field_two_rd.ParseFromString(encoded));
    EXPECT_FALSE(field_two_rd.has_args());
  }

  Trace trace;
  Device device;
  device.set_name("/device:TPU:0");
  (*device.mutable_resources())[1].set_name("XLA Ops");
  (*device.mutable_resources())[2].set_name("Regular Stream");
  (*device.mutable_resources())[3].set_name("Data Motion Layers Utilization");
  (*trace.mutable_devices())[0] = device;

  TestTagFiveTraceEventsContainer container(trace);

  // 1. XLA Ops track.
  TraceEvent xla_ev;
  xla_ev.set_name("xla_wrapper");
  xla_ev.set_timestamp_ps(1000);
  xla_ev.set_duration_ps(100);
  xla_ev.set_raw_data(MakeTagFiveRawDataWithStrArgs(
      {{"hlo_op", "fusion.5"}, {"hlo_module", "train_mod"}}));
  container.AddCompleteEvent(0, 1, &xla_ev);

  // 2. Regular complete track: one HLO event and one non-HLO event.
  TraceEvent reg_non_hlo;
  reg_non_hlo.set_name("reg_plain");
  reg_non_hlo.set_timestamp_ps(2000);
  reg_non_hlo.set_duration_ps(50);
  reg_non_hlo.set_raw_data(
      MakeTagFiveRawDataWithStrArgs({{"shape", "f32[8,8]"}}));
  container.AddCompleteEvent(0, 2, &reg_non_hlo);

  TraceEvent reg_hlo;
  reg_hlo.set_name("reg_kernel");
  reg_hlo.set_timestamp_ps(2100);
  reg_hlo.set_duration_ps(80);
  reg_hlo.set_raw_data(MakeTagFiveRawDataWithStrArgs(
      {{"hlo_op", "dot.5"}, {"hlo_module", "eval_mod"}, {"program_id", "77"}}));
  container.AddCompleteEvent(0, 2, &reg_hlo);

  // 3. Data-motion layer track.
  TraceEvent dml_ev;
  dml_ev.set_name("dml_ev");
  dml_ev.set_timestamp_ps(3000);
  dml_ev.set_duration_ps(60);
  dml_ev.set_raw_data(MakeTagFiveRawDataWithStrArgs(
      {{"Name", "dml_op_5"}, {"hlo_module", "dml_mod"}}));
  container.AddCompleteEvent(0, 3, &dml_ev);

  // 4. Async track.
  TraceEvent async_ev;
  async_ev.set_timestamp_ps(4000);
  async_ev.set_duration_ps(120);
  async_ev.set_flow_id(901);
  async_ev.set_flow_entry_type(TraceEvent::FLOW_NONE);
  async_ev.set_raw_data(MakeTagFiveRawDataWithStrArgs(
      {{"hlo_op", "async_op_5"}, {"hlo_module", "async_mod_5"}}));
  container.AddAsyncEvent(0, "AsyncTrack", &async_ev);

  // 5. Counter track.
  TraceEvent counter_ev;
  counter_ev.set_timestamp_ps(5000);
  TagFiveRawData counter_rd;
  TraceEventArguments::Argument* counter_arg =
      counter_rd.mutable_args()->add_arg();
  counter_arg->set_name("GiB");
  counter_arg->set_double_value(18.25);
  counter_ev.set_raw_data(counter_rd.SerializeAsString());
  container.AddCounterEvent(0, "HbmUsage", &counter_ev);

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  ASSERT_OK_AND_ASSIGN(
      const std::string compressed,
      ConvertTraceDataToCompressedDeltaSeriesProto(options, container));
  ASSERT_OK_AND_ASSIGN(const xprof::TraceDataResponse response,
                       DecompressAndParseResponse(compressed));

  auto find_complete_series =
      [&](uint32_t tid) -> const xprof::TraceEventSeries* {
    for (const xprof::TraceEventSeries& series : response.complete_events()) {
      if (series.metadata().thread_id() == tid) return &series;
    }
    return nullptr;
  };

  const xprof::TraceEventSeries* xla_series = find_complete_series(1);
  ASSERT_NE(xla_series, nullptr);
  ASSERT_EQ(xla_series->event_metadata_size(), 1);
  EXPECT_THAT(xla_series->event_metadata_indices(), ElementsAre(0));
  EXPECT_EQ(
      response.interned_strings(xla_series->event_metadata(0).hlo_op_ref()),
      "fusion.5");
  EXPECT_EQ(
      response.interned_strings(xla_series->event_metadata(0).hlo_module_ref()),
      "train_mod");

  const xprof::TraceEventSeries* reg_series = find_complete_series(2);
  ASSERT_NE(reg_series, nullptr);
  ASSERT_EQ(reg_series->event_metadata_size(), 1);
  EXPECT_THAT(reg_series->event_metadata_indices(), ElementsAre(1));
  EXPECT_EQ(
      response.interned_strings(reg_series->event_metadata(0).hlo_op_ref()),
      "dot.5");
  EXPECT_EQ(
      response.interned_strings(reg_series->event_metadata(0).hlo_module_ref()),
      "eval_mod(77)");

  const xprof::TraceEventSeries* dml_series = find_complete_series(3);
  ASSERT_NE(dml_series, nullptr);
  ASSERT_EQ(dml_series->event_metadata_size(), 1);
  EXPECT_THAT(dml_series->event_metadata_indices(), ElementsAre(0));
  EXPECT_EQ(
      response.interned_strings(dml_series->event_metadata(0).hlo_op_ref()),
      "dml_op_5");
  EXPECT_EQ(
      response.interned_strings(dml_series->event_metadata(0).hlo_module_ref()),
      "dml_mod");

  ASSERT_EQ(response.async_events_size(), 1);
  const xprof::TraceEventSeries& async_series = response.async_events(0);
  ASSERT_EQ(async_series.event_metadata_size(), 1);
  EXPECT_THAT(async_series.event_metadata_indices(), ElementsAre(0));
  EXPECT_EQ(async_series.event_metadata(0).flow_id(), 0u);
  EXPECT_EQ(
      response.interned_strings(async_series.event_metadata(0).hlo_op_ref()),
      "async_op_5");
  EXPECT_EQ(response.interned_strings(
                async_series.event_metadata(0).hlo_module_ref()),
            "async_mod_5");

  ASSERT_EQ(response.counter_events_size(), 1);
  const xprof::TraceEventSeries& counter_series = response.counter_events(0);
  EXPECT_EQ(
      response.interned_strings(counter_series.metadata().event_stats_ref()),
      "GiB");
  ASSERT_EQ(counter_series.event_metadata_size(), 1);
  EXPECT_DOUBLE_EQ(counter_series.event_metadata(0).counter_value_double(),
                   18.25);
}

TEST(DeltaSeriesProtoConverterTest,
     DefaultRawDataArgsExtractorFallbackWorksWhenOmitted) {
  Trace trace;
  Device device;
  device.set_name("/device:TPU:0");
  (*device.mutable_resources())[1].set_name("XLA Ops");
  (*trace.mutable_devices())[0] = device;

  TestTraceEventsContainer container(trace);
  TraceEvent ev;
  ev.set_name("wrapper");
  ev.set_timestamp_ps(1000);
  ev.set_duration_ps(100);
  ev.set_raw_data(MakeRawDataWithStrArgs(
      {{"hlo_op", "default_op"}, {"hlo_module", "default_mod"}}));
  container.AddCompleteEvent(0, 1, &ev);

  DeltaSeriesProtoConversionOptions options;
  options.sparse_event_metadata = true;
  DeltaSeriesProtoConverter converter(
      &container.trace(), internal::MakeDefaultCounterExtractor<RawData>(),
      options);
  xprof::TraceDataResponse response;
  ASSERT_THAT(std::move(converter).GenerateResponse(container, &response),
              IsOk());

  ASSERT_EQ(response.complete_events_size(), 1);
  ASSERT_EQ(response.complete_events(0).event_metadata_size(), 1);
  EXPECT_EQ(response.interned_strings(
                response.complete_events(0).event_metadata(0).hlo_op_ref()),
            "default_op");
  EXPECT_EQ(response.interned_strings(
                response.complete_events(0).event_metadata(0).hlo_module_ref()),
            "default_mod");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
