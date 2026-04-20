#include "frontend/app/components/trace_viewer_v2/trace_helper/perfetto_proto_parser.h"

#include <string>
#include <vector>

#include "<gtest/gtest.h>"
#include "protos/perfetto/trace/trace.pb.h"
#include "protos/perfetto/trace/trace_packet.pb.h"
#include "protos/perfetto/trace/track_event/track_descriptor.pb.h"
#include "protos/perfetto/trace/track_event/track_event.pb.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

TEST(PerfettoEventParserTest, BasicSlice) {
  perfetto::protos::Trace trace;

  // Packet 1: TrackDescriptor (Thread)
  {
    auto* packet = trace.add_packet();
    packet->set_timestamp(0);
    auto* desc = packet->mutable_track_descriptor();
    desc->set_uuid(10);
    auto* thread = desc->mutable_thread();
    thread->set_pid(1);
    thread->set_tid(2);
    thread->set_thread_name("TestThread");
  }

  // Packet 1.5: TrackDescriptor (Process)
  {
    auto* packet = trace.add_packet();
    packet->set_timestamp(0);
    auto* desc = packet->mutable_track_descriptor();
    desc->set_uuid(20);
    auto* process = desc->mutable_process();
    process->set_pid(1);
    process->set_process_name("TestProcess");
  }

  // Packet 2: Slice Begin
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(1000);  // 1000 ns = 1 us
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
    event->set_name("TestSlice");
  }

  // Packet 3: Slice End
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(3000);  // 3000 ns = 3 us
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
  }

  std::string serialized_trace;
  ASSERT_TRUE(trace.SerializeToString(&serialized_trace));

  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(serialized_trace);
  EXPECT_FALSE(parsed_events.parsing_failed);
  ASSERT_GE(parsed_events.flame_events.size(), 1);

  const TraceEvent* slice_event = nullptr;
  for (const auto& event : parsed_events.flame_events) {
    if (event.name == "TestSlice") {
      slice_event = &event;
      break;
    }
  }

  ASSERT_NE(slice_event, nullptr);
  EXPECT_EQ(slice_event->ts, 1.0);   // 1000 ns = 1 us
  EXPECT_EQ(slice_event->dur, 2.0);  // 3000 - 1000 = 2000 ns = 2 us
  EXPECT_EQ(slice_event->pid, 1);
  EXPECT_EQ(slice_event->tid, 2);
}

TEST(PerfettoEventParserTest, InternedStringsAndIncrementalState) {
  perfetto::protos::Trace trace;

  // Packet 1: TrackDescriptor (Thread)
  {
    auto* packet = trace.add_packet();
    packet->set_timestamp(0);
    auto* desc = packet->mutable_track_descriptor();
    desc->set_uuid(10);
    auto* thread = desc->mutable_thread();
    thread->set_pid(1);
    thread->set_tid(2);
    thread->set_thread_name("TestThread");
  }

  // Packet 2: Interned String "FirstInterned" with iid 1
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(100);
    auto* interned_data = packet->mutable_interned_data();
    auto* event_name = interned_data->add_event_names();
    event_name->set_iid(1);
    event_name->set_name("FirstInterned");
  }

  // Packet 3: Slice using Interned String iid 1
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(200);  // 0.2 us
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
    event->set_name_iid(1);
  }
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(300);  // 0.3 us
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
  }

  // Packet 4: Incremental State Cleared
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(400);
    packet->set_incremental_state_cleared(true);
  }

  // Packet 5: Interned String "SecondInterned" with iid 1 (after clear)
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(500);
    auto* interned_data = packet->mutable_interned_data();
    auto* event_name = interned_data->add_event_names();
    event_name->set_iid(1);
    event_name->set_name("SecondInterned");
  }

  // Packet 6: Slice using Interned String iid 1 (should now be
  // "SecondInterned")
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(600);  // 0.6 us
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
    event->set_name_iid(1);
  }
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_timestamp(700);  // 0.7 us
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
  }

  std::string serialized_trace;
  ASSERT_TRUE(trace.SerializeToString(&serialized_trace));

  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(serialized_trace);
  EXPECT_FALSE(parsed_events.parsing_failed);
  ASSERT_EQ(parsed_events.flame_events.size(), 3);

  // First slice should use "FirstInterned"
  const TraceEvent& slice1 = parsed_events.flame_events[1];
  EXPECT_EQ(slice1.name, "FirstInterned");
  EXPECT_EQ(slice1.ts, 0.2);
  EXPECT_NEAR(slice1.dur, 0.1, 1e-9);

  // Second slice should use "SecondInterned" due to incremental state clear
  const TraceEvent& slice2 = parsed_events.flame_events[2];
  EXPECT_EQ(slice2.name, "SecondInterned");
  EXPECT_EQ(slice2.ts, 0.6);
  EXPECT_NEAR(slice2.dur, 0.1, 1e-9);
}

TEST(PerfettoEventParserTest, ParseFail) {
  std::string serialized_trace = "invalid trace";
  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(serialized_trace);
  EXPECT_TRUE(parsed_events.parsing_failed);
}

TEST(PerfettoProtoParserTest, ParsesProcessMetadata) {
  perfetto::protos::Trace trace;
  auto* packet = trace.add_packet();
  auto* desc = packet->mutable_track_descriptor();
  auto* process = desc->mutable_process();
  process->set_pid(123);
  process->set_process_name("test_process");

  std::string buffer;

  ASSERT_TRUE(trace.SerializeToString(&buffer));

  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(buffer);

  ASSERT_EQ(parsed_events.flame_events.size(), 1);

  const auto& event = parsed_events.flame_events[0];

  EXPECT_EQ(event.ph, Phase::kMetadata);
  EXPECT_EQ(event.name, std::string(kProcessName));
  EXPECT_EQ(event.pid, 123);

  auto it = event.args.find("name");

  ASSERT_NE(it, event.args.end());
  EXPECT_EQ(it->second, "test_process");
}

TEST(PerfettoProtoParserTest, IncrementalTimestamp) {
  perfetto::protos::Trace trace;

  // Packet 1: TrackDescriptor (Thread)
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    packet->set_incremental_state_cleared(true);
    auto* desc = packet->mutable_track_descriptor();
    desc->set_uuid(10);
    auto* thread = desc->mutable_thread();
    thread->set_pid(1);
    thread->set_tid(2);
    thread->set_thread_name("TestThread");
  }

  // Packet 2: Slice Begin with delta
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
    event->set_name("TestSlice1");
    event->set_timestamp_delta_us(10);  // 10 us = 10000 ns
  }

  // Packet 3: Slice End with delta
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
    event->set_timestamp_delta_us(5);  // +5 us = 15 us total
  }

  // Packet 4: Slice Begin with delta for another slice
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
    event->set_name("TestSlice2");
    event->set_timestamp_delta_us(5);  // +5 us = 20 us total
  }

  // Packet 5: Slice End with delta
  {
    auto* packet = trace.add_packet();
    packet->set_trusted_packet_sequence_id(1);
    auto* event = packet->mutable_track_event();
    event->set_track_uuid(10);
    event->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
    event->set_timestamp_delta_us(10);  // +10 us = 30 us total
  }

  std::string serialized_trace;
  ASSERT_TRUE(trace.SerializeToString(&serialized_trace));

  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(serialized_trace);

  // 1 thread metadata + 2 complete events = 3 flame events
  ASSERT_EQ(parsed_events.flame_events.size(), 3);

  const auto& thread_event = parsed_events.flame_events[0];
  EXPECT_EQ(thread_event.ph, Phase::kMetadata);
  EXPECT_EQ(thread_event.name, std::string(kThreadName));

  const auto& slice1 = parsed_events.flame_events[1];
  EXPECT_EQ(slice1.name, "TestSlice1");
  EXPECT_EQ(slice1.ts, 10.0);
  EXPECT_EQ(slice1.dur, 5.0);

  const auto& slice2 = parsed_events.flame_events[2];
  EXPECT_EQ(slice2.name, "TestSlice2");
  EXPECT_EQ(slice2.ts, 20.0);
  EXPECT_EQ(slice2.dur, 10.0);
}

}  // namespace
}  // namespace traceviewer
