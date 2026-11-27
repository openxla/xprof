#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>
#endif

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
namespace {

using ::testing::Contains;

constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;
struct CompressedTestFrame {
  size_t uncompressed_size = 0;
  std::string compressed;
};

// Serializes and Zstd-compresses a TraceDataResponse for incremental frame
// tests.
absl::StatusOr<std::string> CompressTraceDataResponse(
    const xprof::TraceDataResponse& response,
    size_t* uncompressed_size = nullptr) {
  std::string serialized;
  if (!response.SerializeToString(&serialized)) {
    return absl::InternalError("Failed to serialize TraceDataResponse.");
  }
  if (uncompressed_size != nullptr) {
    *uncompressed_size = serialized.size();
  }
  return tensorflow::profiler::ZstdCompression::Compress(serialized);
}

absl::Status AppendCompressedXChunkFrame(const xprof::TraceDataResponse& frame,
                                         std::string& out) {
  absl::StatusOr<std::string> compressed = CompressTraceDataResponse(frame);
  if (!compressed.ok()) {
    return compressed.status();
  }
  const uint32_t len = static_cast<uint32_t>(compressed->size());
  const char len_bytes[4] = {
      static_cast<char>(len & 0xff),
      static_cast<char>((len >> 8) & 0xff),
      static_cast<char>((len >> 16) & 0xff),
      static_cast<char>((len >> 24) & 0xff),
  };
  out.append(len_bytes, 4);
  out.append(*compressed);
  return absl::OkStatus();
}

// Clears persistent static incremental parser cache state on scope exit so
// tests do not leak cached state to subsequent test cases.
class IncrementalCacheCleanup {
 public:
  IncrementalCacheCleanup() = default;
  IncrementalCacheCleanup(const IncrementalCacheCleanup&) = delete;
  IncrementalCacheCleanup& operator=(const IncrementalCacheCleanup&) = delete;
  ~IncrementalCacheCleanup() { Clear(); }

  void Clear() {
    if (cleared_) {
      return;
    }
    cleared_ = true;
    SetIncrementalStepTimeBudgetForTest(absl::Milliseconds(8));
    SetIncrementalStepClockForTest(nullptr);
    // Starting an initial session on a fresh DataProvider and Timeline triggers
    // ClearPersistentCache() (CancelIncrementalTraceEvents alone only resets
    // transient cursors).
    ColorPalette reset_palette = ColorPalette::Default();
    DataProvider reset_dp;
    Timeline reset_tl(reset_palette);
    BeginIncrementalTraceEvents(std::nullopt, reset_dp, reset_tl);
    CancelIncrementalTraceEvents(reset_dp, reset_tl);
  }

 private:
  bool cleared_ = false;
};

void AddCompleteEventTrack(xprof::TraceDataResponse& frame,
                           xprof::Process& proc, uint32_t tid,
                           absl::string_view thread_name, int num_events) {
  xprof::Thread* thread = proc.add_threads();
  thread->set_id(tid);
  thread->set_name(thread_name);
  xprof::TraceEventSeries* series = frame.add_complete_events();
  series->mutable_metadata()->set_process_id(proc.id());
  series->mutable_metadata()->set_thread_id(tid);
  for (int i = 0; i < num_events; ++i) {
    series->add_deltas(1'000'000'000ULL);
    series->add_durations(500'000'000ULL);
    series->add_name_refs(1);
    series->add_event_metadata();
  }
}

// Builds, serializes, and Zstd-compresses a single-event TraceDataResponse.
absl::StatusOr<CompressedTestFrame> CompressSingleEventFrame(
    absl::string_view event_name) {
  xprof::TraceDataResponse frame;
  frame.add_interned_strings("");
  frame.add_interned_strings(event_name);
  xprof::Process* proc = frame.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  AddCompleteEventTrack(frame, *proc, /*tid=*/10, "XLA Ops", /*num_events=*/1);

  size_t uncompressed_size = 0;
  absl::StatusOr<std::string> compressed =
      CompressTraceDataResponse(frame, &uncompressed_size);
  if (!compressed.ok()) {
    return compressed.status();
  }
  return CompressedTestFrame{
      .uncompressed_size = uncompressed_size,
      .compressed = *std::move(compressed),
  };
}

// Builds, serializes, and Zstd-compresses a two-track TraceDataResponse so
// incremental unmarshaling and cached-block replay span multiple steps.
absl::StatusOr<std::string> CompressMultiEventFrame(
    absl::string_view event_name, int events_per_track = 10,
    uint64_t full_timespan_end_ps = 20'000'000'000ULL) {
  xprof::TraceDataResponse frame;
  frame.set_full_timespan_start_ps(0ULL);
  frame.set_full_timespan_end_ps(full_timespan_end_ps);
  frame.add_interned_strings("");
  frame.add_interned_strings(event_name);
  xprof::Process* proc = frame.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  AddCompleteEventTrack(frame, *proc, /*tid=*/10, "XLA Ops", events_per_track);
  AddCompleteEventTrack(frame, *proc, /*tid=*/11, "XLA Ops 2",
                        events_per_track);
  return CompressTraceDataResponse(frame);
}

void AppendFramedBytes(std::string& out, absl::string_view payload) {
  const uint32_t len = static_cast<uint32_t>(payload.size());
  const char len_bytes[4] = {
      static_cast<char>(len & 0xff),
      static_cast<char>((len >> 8) & 0xff),
      static_cast<char>((len >> 16) & 0xff),
      static_cast<char>((len >> 24) & 0xff),
  };
  out.append(len_bytes, 4);
  out.append(payload.data(), payload.size());
}

absl::Status AppendFramedResponse(std::string& out,
                                  const xprof::TraceDataResponse& frame) {
  const absl::StatusOr<std::string> compressed =
      CompressTraceDataResponse(frame);
  if (!compressed.ok()) {
    return compressed.status();
  }
  AppendFramedBytes(out, *compressed);
  return absl::OkStatus();
}

}  // namespace

TEST(TracePbEventParserIncrementalTest,
     SkipsCorruptZstdAndInvalidProtobufFrames) {
  using ::testing::ElementsAre;

  IncrementalCacheCleanup cleanup_guard;

  // Frame 1: invalid Zstd bytes (triggers !decompressed_view.ok() branch).
  const std::string bad_zstd = "not valid zstd data";

  // Frame 2: valid Zstd frame containing invalid protobuf bytes (triggers
  // !response->ParseFromString(*decompressed_view) branch).
  const absl::StatusOr<std::string> bad_proto_compressed =
      tensorflow::profiler::ZstdCompression::Compress("\x08");
  ASSERT_TRUE(bad_proto_compressed.ok());

  // Frame 3: valid compressed TraceDataResponse to confirm incremental
  // decoding continues cleanly after skipping corrupt frames.
  const absl::StatusOr<CompressedTestFrame> valid_frame =
      CompressSingleEventFrame("valid_after_corrupt");
  ASSERT_TRUE(valid_frame.ok());

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  AppendFramedBytes(xchn_payload, bad_zstd);
  AppendFramedBytes(xchn_payload, *bad_proto_compressed);
  AppendFramedBytes(xchn_payload, valid_frame->compressed);

  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());
  while (StepIncrementalTraceEvents(100, data_provider, timeline)) {
  }
  FinishIncrementalTraceEvents(data_provider, timeline);

  ASSERT_THAT(timeline.timeline_data().entry_names,
              ElementsAre("valid_after_corrupt"));
}

TEST(TracePbEventParserIncrementalTest,
     ResetAndClearPersistentCacheReleasesZstdDecompressorBuffer) {
  constexpr size_t kDefaultCapacity =
      tensorflow::profiler::ZstdDecompressorContext::kDefaultBufferSize;

  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  // Step 1: Sub-default frame (<= kDefaultCapacity) stays reusable across
  // frames and releases back to 0 on FinishIncrementalTraceEvents
  // (session.Reset()).
  const absl::StatusOr<CompressedTestFrame> sub_default_frame =
      CompressSingleEventFrame(std::string(64 * 1024, 's'));
  ASSERT_TRUE(sub_default_frame.ok());
  ASSERT_LE(sub_default_frame->uncompressed_size, kDefaultCapacity);

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(sub_default_frame->compressed.data()),
      sub_default_frame->compressed.size());
  StepIncrementalTraceEvents(1, data_provider, timeline);
  ASSERT_GT(GetZstdDecompressorCapacityForTest(), 0u);
  ASSERT_LE(GetZstdDecompressorCapacityForTest(), kDefaultCapacity);

  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_EQ(GetZstdDecompressorCapacityForTest(), 0u);

  // Step 2: Sub-default frame (<= kDefaultCapacity) releases back to 0 on
  // CancelIncrementalTraceEvents (session.Reset()).
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(sub_default_frame->compressed.data()),
      sub_default_frame->compressed.size());
  StepIncrementalTraceEvents(1, data_provider, timeline);
  ASSERT_GT(GetZstdDecompressorCapacityForTest(), 0u);

  CancelIncrementalTraceEvents(data_provider, timeline);
  EXPECT_EQ(GetZstdDecompressorCapacityForTest(), 0u);

  // Step 3: Sub-default frame (<= kDefaultCapacity) releases back to 0 when
  // ParseAndProcessCompressedTraceEvents calls ClearPersistentCache().
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(sub_default_frame->compressed.data()),
      sub_default_frame->compressed.size());
  StepIncrementalTraceEvents(1, data_provider, timeline);
  ASSERT_GT(GetZstdDecompressorCapacityForTest(), 0u);

  xprof::TraceDataResponse small_frame;
  std::string serialized_small;
  ASSERT_TRUE(small_frame.SerializeToString(&serialized_small));
  const absl::StatusOr<std::string> compressed_small =
      tensorflow::profiler::ZstdCompression::Compress(serialized_small);
  ASSERT_TRUE(compressed_small.ok());

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_small->data()),
      compressed_small->size(), std::nullopt, data_provider, timeline);
  EXPECT_EQ(GetZstdDecompressorCapacityForTest(), 0u);

  // Step 4: Oversized frame (> kDefaultCapacity) releases immediately after
  // ParseFromString (and on truncated-body decode failure) without waiting for
  // FinishIncrementalTraceEvents or CancelIncrementalTraceEvents.
  const absl::StatusOr<CompressedTestFrame> large_frame =
      CompressSingleEventFrame(std::string(kDefaultCapacity + 1024, 'x'));
  ASSERT_TRUE(large_frame.ok());
  ASSERT_GT(large_frame->uncompressed_size, kDefaultCapacity);

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(large_frame->compressed.data()),
      large_frame->compressed.size());
  StepIncrementalTraceEvents(1, data_provider, timeline);
  EXPECT_EQ(GetZstdDecompressorCapacityForTest(), 0u);
  FinishIncrementalTraceEvents(data_provider, timeline);

  ASSERT_GT(large_frame->compressed.size(), 8u);
  const absl::string_view truncated_large(large_frame->compressed.data(),
                                          large_frame->compressed.size() - 5);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(truncated_large.data()),
      truncated_large.size());
  StepIncrementalTraceEvents(1, data_provider, timeline);
  EXPECT_EQ(GetZstdDecompressorCapacityForTest(), 0u);
  CancelIncrementalTraceEvents(data_provider, timeline);
}

TEST(TracePbEventParserIncrementalTest,
     IncrementalRefetchRetainsSparseCoreTecSubTracks) {
  using ::testing::AllOf;
  using ::testing::Contains;
  using ::testing::Field;

  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse initial_frame;
  initial_frame.add_interned_strings("");
  initial_frame.add_interned_strings("tec0_op");
  initial_frame.add_interned_strings("tec1_op");
  initial_frame.set_full_timespan_start_ps(0ULL);
  initial_frame.set_full_timespan_end_ps(100000000000ULL);  // 100 ms

  xprof::Process* proc = initial_frame.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("/device:TPU:0 SparseCore 0");
  // Process must be expanded (default for <5 processes) and have >=2 TEC
  // threads so DataProvider::PopulateTecTracks creates the "TEC Tracks"
  // container at kThreadNestingLevel and TEC 0..1 at kSubTrackNestingLevel.
  xprof::Thread* th0 = proc->add_threads();
  th0->set_id(85);
  th0->set_name("TEC 0");
  xprof::Thread* th1 = proc->add_threads();
  th1->set_id(86);
  th1->set_name("TEC 1");

  const auto populate_tec_series = [](xprof::TraceEventSeries* series,
                                      uint32_t pid, uint32_t tid,
                                      uint32_t name_ref) {
    series->mutable_metadata()->set_process_id(pid);
    series->mutable_metadata()->set_thread_id(tid);
    for (int i = 0; i < 20; ++i) {
      // Events spaced every 1 ms (1,000,000,000 ps), duration 0.5 ms.
      series->add_deltas(1000000000ULL);
      series->add_durations(500000000ULL);
      series->add_name_refs(name_ref);
      series->add_event_metadata();
    }
  };
  populate_tec_series(initial_frame.add_complete_events(), 1, 85, 1);
  populate_tec_series(initial_frame.add_complete_events(), 1, 86, 2);

  std::string serialized;
  ASSERT_TRUE(initial_frame.SerializeToString(&serialized));
  const absl::StatusOr<std::string> compressed =
      tensorflow::profiler::ZstdCompression::Compress(serialized);
  ASSERT_TRUE(compressed.ok());

  // Set threshold low enough that initial overview load arms the in-memory
  // block cache.
  timeline.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed->data()), compressed->size());
  while (StepIncrementalTraceEvents(5000, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  // Verify the initial group hierarchy created "TEC Tracks" (collapsed by
  // default) and leaf sub-tracks "TEC 0" and "TEC 1" at kSubTrackNestingLevel.
  ASSERT_THAT(
      timeline.timeline_data().groups,
      AllOf(
          Contains(AllOf(Field("name", &Group::name, kTecTracksGroupName),
                         Field("nesting_level", &Group::nesting_level,
                               kThreadNestingLevel),
                         Field("has_children", &Group::has_children, true),
                         Field("expanded", &Group::expanded, false))),
          Contains(AllOf(Field("name", &Group::name, "TEC 0"),
                         Field("nesting_level", &Group::nesting_level,
                               kSubTrackNestingLevel),
                         Field("has_children", &Group::has_children, false))),
          Contains(AllOf(Field("name", &Group::name, "TEC 1"),
                         Field("nesting_level", &Group::nesting_level,
                               kSubTrackNestingLevel),
                         Field("has_children", &Group::has_children, false)))));

  // 1. Zoom-in cached refetch over [0.5, 5.5] ms while "TEC Tracks" is
  // collapsed: both TEC 0 and TEC 1 events in the window must be retained so
  // the collapsed "TEC Tracks" preview row stays populated.
  BeginIncrementalTraceEvents(std::make_pair(0.5, 5.5), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(5000, data_provider, timeline)) {
  }
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_THAT(timeline.timeline_data().entry_names,
              Contains("tec0_op").Times(5));
  EXPECT_THAT(timeline.timeline_data().entry_names,
              Contains("tec1_op").Times(5));

  // 2. Expand "TEC Tracks" (via RevealEvent) and perform another zoom-in
  // cached refetch over [1.6, 4.4] ms: visible leaf sub-tracks TEC 0 and TEC 1
  // at kSubTrackNestingLevel must remain in active_track_filter and retain
  // their in-window events (starting at 2.0, 3.0, and 4.0 ms).
  timeline.RevealEvent(0);
  ASSERT_THAT(timeline.timeline_data().groups,
              Contains(AllOf(Field("name", &Group::name, kTecTracksGroupName),
                             Field("expanded", &Group::expanded, true))));

  BeginIncrementalTraceEvents(std::make_pair(1.6, 4.4), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(5000, data_provider, timeline)) {
  }
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_THAT(timeline.timeline_data().entry_names,
              Contains("tec0_op").Times(3));
  EXPECT_THAT(timeline.timeline_data().entry_names,
              Contains("tec1_op").Times(3));

  cleanup_guard.Clear();
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
}

TEST(TracePbEventParserIncrementalTest,
     HasFlushedIncrementalTraceEventsTracksFirstTimelineFlushAndResets) {
  using ::testing::Contains;
  using ::testing::Field;

  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  // 1. Metadata-only frame (timespan/details/interned_strings plus
  // metadata.processes and threads, but no complete/async/counter events)
  // appends Phase::kMetadata batch events without flushing the timeline.
  xprof::TraceDataResponse metadata_only;
  metadata_only.set_full_timespan_start_ps(0ULL);
  metadata_only.set_full_timespan_end_ps(10'000'000'000ULL);
  metadata_only.add_interned_strings("");
  auto* detail = metadata_only.add_details();
  detail->set_name("mpmd_pipeline_view");
  detail->set_value(false);
  xprof::Process* metadata_proc =
      metadata_only.mutable_metadata()->add_processes();
  metadata_proc->set_id(1);
  metadata_proc->set_name("metadata_only_process");
  xprof::Thread* metadata_thread = metadata_proc->add_threads();
  metadata_thread->set_id(10);
  metadata_thread->set_name("XLA Ops");
  const absl::StatusOr<std::string> c_metadata =
      CompressTraceDataResponse(metadata_only);
  ASSERT_TRUE(c_metadata.ok());

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c_metadata->data()), c_metadata->size());
  EXPECT_FALSE(StepIncrementalTraceEvents(100, data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  EXPECT_TRUE(timeline.timeline_data().groups.empty());

  // 2. Multi-step Frame 1 exceeding max_events_per_step keeps
  // HasFlushedIncrementalTraceEvents() false on step 1 (at_frame_boundary ==
  // false) and sets it true once step 2 reaches the frame boundary and flushes.
  // Process 1 carries no metadata in this frame, so its group name must come
  // from the earlier metadata-only frame.
  xprof::TraceDataResponse multi_event_frame;
  multi_event_frame.add_interned_strings("");
  multi_event_frame.add_interned_strings("op_step");
  xprof::Process* proc = multi_event_frame.mutable_metadata()->add_processes();
  proc->set_id(2);
  proc->set_name("host:0 /device:TPU:1");
  xprof::Thread* thread = proc->add_threads();
  thread->set_id(10);
  thread->set_name("XLA Ops");
  for (uint32_t pid = 1; pid <= 2; ++pid) {
    xprof::TraceEventSeries* series = multi_event_frame.add_complete_events();
    series->mutable_metadata()->set_process_id(pid);
    series->mutable_metadata()->set_thread_id(10);
    for (int i = 0; i < 2; ++i) {
      series->add_deltas(1'000'000'000ULL);
      series->add_durations(500'000'000ULL);
      series->add_name_refs(1);
      series->add_event_metadata();
    }
  }
  const absl::StatusOr<std::string> c_events =
      CompressTraceDataResponse(multi_event_frame);
  ASSERT_TRUE(c_events.ok());

  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c_events->data()), c_events->size());
  EXPECT_TRUE(StepIncrementalTraceEvents(/*max_events_per_step=*/2,
                                         data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());

  EXPECT_FALSE(StepIncrementalTraceEvents(/*max_events_per_step=*/10,
                                          data_provider, timeline));
  EXPECT_TRUE(HasFlushedIncrementalTraceEvents());
  EXPECT_THAT(timeline.timeline_data().groups,
              Contains(Field("name", &Group::name, "metadata_only_process")));

  // 3. FinishIncrementalTraceEvents resets the session and flush flag.
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());

  // 4. CancelIncrementalTraceEvents on a fresh initial load also resets the
  // flush flag.
  const absl::StatusOr<CompressedTestFrame> single_frame =
      CompressSingleEventFrame("cancel_op");
  ASSERT_TRUE(single_frame.ok());
  DataProvider cancel_data_provider;
  Timeline cancel_timeline(palette);
  BeginIncrementalTraceEvents(std::nullopt, cancel_data_provider,
                              cancel_timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(single_frame->compressed.data()),
      single_frame->compressed.size());
  EXPECT_FALSE(StepIncrementalTraceEvents(
      /*max_events_per_step=*/10, cancel_data_provider, cancel_timeline));
  EXPECT_TRUE(HasFlushedIncrementalTraceEvents());
  CancelIncrementalTraceEvents(cancel_data_provider, cancel_timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
}

TEST(TracePbEventParserIncrementalTest,
     HasFlushedIncrementalTraceEventsStaysFalseOnSubsequentUnfilteredSession) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  const absl::StatusOr<std::string> compressed =
      CompressMultiEventFrame("unfiltered_refetch_op");
  ASSERT_TRUE(compressed.ok());

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed->data()), compressed->size(),
      std::nullopt, data_provider, timeline);
  ASSERT_TRUE(data_provider.HasKnownTracks());
  ASSERT_FALSE(HasCachedTraceEventsIncremental());

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed->data()), compressed->size());
  EXPECT_TRUE(StepIncrementalTraceEvents(/*max_events_per_step=*/5,
                                         data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  EXPECT_FALSE(StepIncrementalTraceEvents(/*max_events_per_step=*/20,
                                          data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
}

TEST(TracePbEventParserIncrementalTest,
     HasFlushedIncrementalTraceEventsStaysFalseOnFilteredUncachedSession) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  const absl::StatusOr<std::string> compressed =
      CompressMultiEventFrame("filtered_refetch_op");
  ASSERT_TRUE(compressed.ok());

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed->data()), compressed->size(),
      std::nullopt, data_provider, timeline);
  ASSERT_TRUE(data_provider.HasKnownTracks());
  ASSERT_FALSE(HasCachedTraceEventsIncremental());

  BeginIncrementalTraceEvents(std::make_pair(0.0, 10.0), data_provider,
                              timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed->data()), compressed->size());
  EXPECT_TRUE(StepIncrementalTraceEvents(/*max_events_per_step=*/5,
                                         data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  EXPECT_FALSE(StepIncrementalTraceEvents(/*max_events_per_step=*/20,
                                          data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
}

TEST(TracePbEventParserIncrementalTest,
     HasFlushedIncrementalTraceEventsStaysFalseOnCachedStoreReplay) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  const absl::StatusOr<std::string> compressed =
      CompressMultiEventFrame("cached_op");
  ASSERT_TRUE(compressed.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(8);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed->data()), compressed->size());
  EXPECT_FALSE(StepIncrementalTraceEvents(/*max_events_per_step=*/50,
                                          data_provider, timeline));
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());

  BeginIncrementalTraceEvents(std::make_pair(1.0, 8.0), data_provider,
                              timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  EXPECT_TRUE(StepIncrementalTraceEvents(/*max_events_per_step=*/4,
                                         data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  EXPECT_FALSE(StepIncrementalTraceEvents(/*max_events_per_step=*/20,
                                          data_provider, timeline));
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_FALSE(HasFlushedIncrementalTraceEvents());
}

TEST(TracePbEventParserIncrementalTest,
     CachedRefetchBinsSubPixelLeafRectsAndEmitsRawOnDeepZoom) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("wide_parent");
  f0.add_interned_strings("tiny_leaf");
  f0.add_interned_strings("flow_leaf");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(1000000000000ULL);  // 1000 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);

  // Wide parent rect covering [0, 50] ms (50,000 us).
  series->add_deltas(0ULL);
  series->add_durations(50000000000ULL);
  series->add_name_refs(1);
  series->add_event_metadata();

  // 20,000 tiny 1 us leaf rects spaced 2 us apart across [0, 40] ms, with
  // flow-tagged leaves at indices 500 and 1500 that must never be coalesced.
  constexpr int kTinyEvents = 20000;
  for (int i = 0; i < kTinyEvents; ++i) {
    series->add_deltas(i == 0 ? 1000000ULL : 2000000ULL);  // +2 us each
    series->add_durations(1000000ULL);                     // 1 us
    auto* meta = series->add_event_metadata();
    if (i == 500 || i == 1500) {
      series->add_name_refs(3);
      meta->set_flow_id(42);
    } else {
      series->add_name_refs(2);
    }
  }

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c0->data()), c0->size());
  while (StepIncrementalTraceEvents(65536, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  // 1. Wide local refetch over [0, 800] ms (min_bin ~= 97.6 us >> 1 us):
  // sub-pixel leaf bursts coalesce down to ~400 bins, while the wide parent
  // and flow endpoints stay intact and window_at_full_resolution() stays true.
  BeginIncrementalTraceEvents(std::make_pair(0.0, 800.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(65536, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  EXPECT_TRUE(timeline.window_at_full_resolution());
  const size_t wide_count = timeline.timeline_data().entry_names.size();
  EXPECT_GT(wide_count, 10u);
  EXPECT_LT(wide_count, 2000u);
  EXPECT_TRUE(timeline.timeline_data().flow_lines_by_flow_id.contains("42"));
  bool found_wide_parent = false;
  for (const std::string& name : timeline.timeline_data().entry_names) {
    if (name == "wide_parent") {
      found_wide_parent = true;
      break;
    }
  }
  EXPECT_TRUE(found_wide_parent);

  // 2. Deep local refetch over [0, 4] ms (min_bin ~= 0.49 us < 1 us): all raw
  // leaf rects inside [0, 4] ms emit unbinned (1 wide_parent + 2000 tiny).
  BeginIncrementalTraceEvents(std::make_pair(0.0, 4.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(65536, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  EXPECT_EQ(timeline.timeline_data().entry_names.size(), 2001u);
}

TEST(TracePbEventParserIncrementalTest,
     CachedRefetchSkipsOutOfRangeSplitBlocks) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("early_split_op");
  f0.add_interned_strings("late_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(100000000000ULL);  // 100 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  // Early series block containing only split-phase events in [1, 2] ms.
  auto* early = f0.add_complete_events();
  early->mutable_metadata()->set_process_id(1);
  early->mutable_metadata()->set_thread_id(10);
  early->add_deltas(1000000000ULL);
  early->add_durations(500000000ULL | kDurationPhaseBeginBit);
  early->add_name_refs(1);
  early->add_event_metadata();
  early->add_deltas(0ULL);
  early->add_durations(1000000000ULL | kDurationPhaseEndBit);
  early->add_name_refs(1);
  early->add_event_metadata();

  // Late series block containing 64 normal events in [50, 56.4] ms.
  auto* late = f0.add_complete_events();
  late->mutable_metadata()->set_process_id(1);
  late->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 64; ++i) {
    late->add_deltas(i == 0 ? 50000000000ULL : 100000000ULL);
    late->add_durations(50000000ULL);
    late->add_name_refs(2);
    late->add_event_metadata();
  }

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c0->data()), c0->size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  // Refetch [49, 60] ms: early split block [1, 2] ms is pruned by bounds and
  // only late_op events are decoded.
  BeginIncrementalTraceEvents(std::make_pair(49.0, 60.0), data_provider,
                              timeline);
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_EQ(timeline.timeline_data().entry_names.size(), 64u);
  for (const std::string& name : timeline.timeline_data().entry_names) {
    EXPECT_EQ(name, "late_op");
  }
}

TEST(TracePbEventParserIncrementalTest,
     CachedRefetchRespectsRawScannedEventBudgetPerStep) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("tiny_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000000ULL);  // 10,000 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  // 270,000 tiny 1 us rects (> 262,144 raw-scanned cap per step) that bin down
  // to far fewer than 131,072 emitted rects across [0, 8000] ms.
  constexpr int kTotalEvents = 270000;
  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);
  series->mutable_deltas()->Reserve(kTotalEvents);
  series->mutable_durations()->Reserve(kTotalEvents);
  series->mutable_name_refs()->Reserve(kTotalEvents);
  series->mutable_event_metadata()->Reserve(kTotalEvents);
  for (int i = 0; i < kTotalEvents; ++i) {
    series->add_deltas(i == 0 ? 1000000ULL : 2000000ULL);  // +2 us each
    series->add_durations(1000000ULL);                     // 1 us
    series->add_name_refs(1);
    series->add_event_metadata();
  }

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c0->data()), c0->size());
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  BeginIncrementalTraceEvents(std::make_pair(0.0, 8000.0), data_provider,
                              timeline);
  // First step must yield before decoding all 270,000 raw events even though
  // binned output count is well below 131,072.
  EXPECT_TRUE(StepIncrementalTraceEvents(131072, data_provider, timeline));
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  EXPECT_GT(timeline.timeline_data().entry_names.size(), 0u);
  EXPECT_LT(timeline.timeline_data().entry_names.size(), 2000u);
}

TEST(TracePbEventParserNativeTest,
     LocalReplayCoalescingPreservesIdleGapsAndDetailLookups) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("isolated_a");
  f0.add_interned_strings("isolated_b");
  f0.add_interned_strings("burst_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000000ULL);  // 10,000 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);

  // Two 10 us leaf rects separated by a 500 us idle gap within the same ~976 us
  // bin of an [0, 8000] ms replay window must stay separate and retain serials.
  series->add_deltas(100000000ULL);    // 100 us
  series->add_durations(10000000ULL);  // 10 us
  series->add_name_refs(1);
  series->add_event_metadata()->set_serial(101);

  series->add_deltas(510000000ULL);    // 610 us (500 us idle gap)
  series->add_durations(10000000ULL);  // 10 us
  series->add_name_refs(2);
  series->add_event_metadata()->set_serial(202);

  // Ten tightly packed 1 us leaf rects (0.2 us gap) in a later bin coalesce
  // into a single synthetic burst without a stale single-event uid.
  for (int i = 0; i < 10; ++i) {
    series->add_deltas(i == 0 ? 1390000000ULL : 1200000ULL);  // 2000 us + 1.2us
    series->add_durations(1000000ULL);                        // 1 us
    series->add_name_refs(3);
    series->add_event_metadata()->set_serial(300 + i);
  }
  // Pad past max_uncompacted_overview_entries so block cache is built.
  for (int i = 0; i < 20; ++i) {
    series->add_deltas(10000000000ULL);  // +10 ms each
    series->add_durations(1000000ULL);
    series->add_name_refs(3);
    series->add_event_metadata()->set_serial(400 + i);
  }

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c0->data()), c0->size());
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  BeginIncrementalTraceEvents(std::make_pair(0.0, 8000.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));

  const auto& td = timeline.timeline_data();
  int found_a = -1;
  int found_b = -1;
  int found_burst = -1;
  for (size_t i = 0; i < td.entry_names.size(); ++i) {
    if (td.entry_names[i] == "isolated_a") found_a = static_cast<int>(i);
    if (td.entry_names[i] == "isolated_b") found_b = static_cast<int>(i);
    if (td.entry_names[i] == "burst_op" && td.entry_start_times[i] >= 1999.0 &&
        td.entry_start_times[i] <= 2001.0) {
      found_burst = static_cast<int>(i);
    }
  }
  ASSERT_GE(found_a, 0);
  ASSERT_GE(found_b, 0);
  ASSERT_GE(found_burst, 0);
  EXPECT_DOUBLE_EQ(td.entry_total_times[found_a], 10.0);
  EXPECT_DOUBLE_EQ(td.entry_total_times[found_b], 10.0);
  EXPECT_EQ(td.GetEntryArgs(found_a).at("uid"), "101");
  EXPECT_EQ(td.GetEntryArgs(found_b).at("uid"), "202");
  EXPECT_GT(td.entry_total_times[found_burst], 10.0);
  EXPECT_FALSE(td.GetEntryArgs(found_burst).contains("uid"));
}

TEST(TracePbEventParserNativeTest,
     LocalReplayDisablesBinningForNarrowWidenedWindows) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("narrow_leaf");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(800000000000ULL);  // 800 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);

  // Twenty tightly packed 1 us leaf rects (0.2 us gap) inside [1.0, 1.024] ms.
  // On a 42 ms widened deep-zoom replay window (<= 100 ms), binning must be
  // disabled so every rect emits raw with its original uid intact; on an 800 ms
  // wide replay (> 100 ms), they still coalesce into a synthetic burst.
  for (int i = 0; i < 20; ++i) {
    series->add_deltas(i == 0 ? 1000000000ULL : 1200000ULL);  // 1 ms + 1.2 us
    series->add_durations(1000000ULL);                        // 1 us
    series->add_name_refs(1);
    series->add_event_metadata()->set_serial(500 + i);
  }
  // Pad past max_uncompacted_overview_entries so block cache is built.
  for (int i = 0; i < 20; ++i) {
    series->add_deltas(10000000000ULL);  // +10 ms each
    series->add_durations(1000000ULL);
    series->add_name_refs(1);
    series->add_event_metadata()->set_serial(600 + i);
  }

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c0->data()), c0->size());
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  // 1. Widened narrow replay window [0, 42] ms (<= 100 ms): binning disabled;
  // all 20 tight 1 us rects inside [1, 1.024] ms emit raw with uid intact.
  BeginIncrementalTraceEvents(std::make_pair(0.0, 42.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  const auto& narrow_td = timeline.timeline_data();
  int raw_cluster_count = 0;
  for (size_t i = 0; i < narrow_td.entry_names.size(); ++i) {
    if (narrow_td.entry_start_times[i] >= 999.0 &&
        narrow_td.entry_start_times[i] <= 1100.0) {
      ++raw_cluster_count;
      EXPECT_DOUBLE_EQ(narrow_td.entry_total_times[i], 1.0);
      EXPECT_TRUE(narrow_td.GetEntryArgs(i).contains("uid"));
    }
  }
  EXPECT_EQ(raw_cluster_count, 20);

  // 2. Wide replay window [0, 800] ms (> 100 ms): binning stays active and
  // collapses the 20 tight 1 us rects into a single synthetic burst.
  BeginIncrementalTraceEvents(std::make_pair(0.0, 800.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  const auto& wide_td = timeline.timeline_data();
  int wide_cluster_count = 0;
  for (size_t i = 0; i < wide_td.entry_names.size(); ++i) {
    if (wide_td.entry_start_times[i] >= 999.0 &&
        wide_td.entry_start_times[i] <= 1100.0) {
      ++wide_cluster_count;
      EXPECT_GT(wide_td.entry_total_times[i], 1.0);
      EXPECT_FALSE(wide_td.GetEntryArgs(i).contains("uid"));
    }
  }
  EXPECT_EQ(wide_cluster_count, 1);
}

TEST(TracePbEventParserNativeTest,
     LocalReplayPreservesShortNonLeafParentHierarchy) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("short_parent");
  f0.add_interned_strings("child_a");
  f0.add_interned_strings("child_b");
  f0.add_interned_strings("sibling_after");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000000ULL);

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);

  // Non-leaf parent [100, 200] us (< ~976 us min_bin_ps for [0, 8000] ms)
  // enclosing two children [110, 115] us and [116, 120] us, followed closely by
  // sibling_after [201, 205] us in the same bin.
  series->add_deltas(100000000ULL);     // 100 us
  series->add_durations(100000000ULL);  // 100 us -> ends at 200 us
  series->add_name_refs(1);
  series->add_event_metadata()->set_serial(1);

  series->add_deltas(10000000ULL);    // 110 us
  series->add_durations(5000000ULL);  // 5 us
  series->add_name_refs(2);
  series->add_event_metadata()->set_serial(2);

  series->add_deltas(6000000ULL);     // 116 us (1 us gap inside parent)
  series->add_durations(4000000ULL);  // 4 us
  series->add_name_refs(3);
  series->add_event_metadata()->set_serial(3);

  series->add_deltas(85000000ULL);    // 201 us (outside parent [100, 200])
  series->add_durations(4000000ULL);  // 4 us
  series->add_name_refs(4);
  series->add_event_metadata()->set_serial(4);

  for (int i = 0; i < 20; ++i) {
    series->add_deltas(10000000000ULL);
    series->add_durations(1000000ULL);
    series->add_name_refs(4);
    series->add_event_metadata()->set_serial(10 + i);
  }

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok());

  timeline.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c0->data()), c0->size());
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  BeginIncrementalTraceEvents(std::make_pair(0.0, 8000.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));

  const auto& td = timeline.timeline_data();
  int parent_idx = -1;
  int child_idx = -1;
  int sibling_idx = -1;
  for (size_t i = 0; i < td.entry_names.size(); ++i) {
    if (td.entry_names[i] == "short_parent") {
      parent_idx = static_cast<int>(i);
    } else if (td.entry_start_times[i] >= 109.5 &&
               td.entry_start_times[i] <= 110.5) {
      child_idx = static_cast<int>(i);
    } else if (td.entry_names[i] == "sibling_after" &&
               td.entry_start_times[i] >= 200.5 &&
               td.entry_start_times[i] <= 201.5) {
      sibling_idx = static_cast<int>(i);
    }
  }
  ASSERT_GE(parent_idx, 0);
  ASSERT_GE(child_idx, 0);
  ASSERT_GE(sibling_idx, 0);
  EXPECT_DOUBLE_EQ(td.entry_total_times[parent_idx], 100.0);
  EXPECT_GT(td.entry_levels[child_idx], td.entry_levels[parent_idx]);
  EXPECT_LE(td.entry_start_times[child_idx] + td.entry_total_times[child_idx],
            200.0);
  EXPECT_EQ(td.entry_levels[sibling_idx], td.entry_levels[parent_idx]);
}

TEST(TracePbEventParserNativeTest,
     LocalReplayPersistsParentStackAcrossBlockAndSplitBoundaries) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  // Frame 1: Track 10 has a parent in Block 0 spanning [100, 10000] us with
  // 4,100 children crossing the 4,096-event CachedCompleteBlock boundary, then
  // a sibling at 10000.5 us right outside the parent end. Track 20 has a split
  // Begin span [1000, 1500] us that closes in Frame 2 at 2500 us.
  xprof::TraceDataResponse frame1;
  frame1.set_full_timespan_start_ps(0ULL);
  frame1.set_full_timespan_end_ps(10000000000000ULL);
  frame1.add_interned_strings("");
  frame1.add_interned_strings("cross_block_parent");
  frame1.add_interned_strings("block_child");
  frame1.add_interned_strings("block_sibling");
  frame1.add_interned_strings("split_parent");
  frame1.add_interned_strings("split_child");
  frame1.add_interned_strings("split_sibling");
  auto* proc = frame1.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th10 = proc->add_threads();
  th10->set_id(10);
  th10->set_name("Track Block");
  auto* th20 = proc->add_threads();
  th20->set_id(20);
  th20->set_name("Track Split");

  auto* s10 = frame1.add_complete_events();
  s10->mutable_metadata()->set_process_id(1);
  s10->mutable_metadata()->set_thread_id(10);
  s10->add_deltas(100000000ULL);      // 100 us
  s10->add_durations(9900000000ULL);  // ends at 10000 us
  s10->add_name_refs(1);
  s10->add_event_metadata()->set_serial(1);
  uint64_t ts_ps = 100000000ULL;
  // 4,100 children: 4,095 in Block 0, 5 in Block 1; last child is [9995, 9999].
  for (int i = 0; i < 4100; ++i) {
    const uint64_t next_ts_ps =
        (i == 4099) ? 9995000000ULL : (ts_ps + 2000000ULL);
    s10->add_deltas(next_ts_ps - ts_ps);
    s10->add_durations(i == 4099 ? 4000000ULL : 1000000ULL);
    s10->add_name_refs(2);
    s10->add_event_metadata()->set_serial(2 + i);
    ts_ps = next_ts_ps;
  }
  // Sibling starts at 10000.2 us (0.2 us after last child, across parent end).
  s10->add_deltas(10000200000ULL - ts_ps);
  s10->add_durations(4000000ULL);
  s10->add_name_refs(3);
  s10->add_event_metadata()->set_serial(5000);

  auto* s20_f1 = frame1.add_complete_events();
  s20_f1->mutable_metadata()->set_process_id(1);
  s20_f1->mutable_metadata()->set_thread_id(20);
  s20_f1->add_deltas(1000000000ULL);
  s20_f1->add_durations(500000000ULL | kDurationPhaseBeginBit);
  s20_f1->add_name_refs(4);
  s20_f1->add_event_metadata()->set_serial(6000);

  xprof::TraceDataResponse frame2;
  auto* s20_f2 = frame2.add_complete_events();
  s20_f2->mutable_metadata()->set_process_id(1);
  s20_f2->mutable_metadata()->set_thread_id(20);
  s20_f2->add_deltas(1000000000ULL);
  s20_f2->add_durations(1500000000ULL | kDurationPhaseEndBit);  // ends 2500 us
  s20_f2->add_name_refs(4);
  s20_f2->add_event_metadata()->set_serial(6000);
  s20_f2->add_deltas(1495000000ULL);  // 2495 us
  s20_f2->add_durations(4000000ULL);  // [2495, 2499] us inside split_parent
  s20_f2->add_name_refs(5);
  s20_f2->add_event_metadata()->set_serial(6001);
  s20_f2->add_deltas(5200000ULL);  // 2500.2 us outside split_parent
  s20_f2->add_durations(4000000ULL);
  s20_f2->add_name_refs(6);
  s20_f2->add_event_metadata()->set_serial(6002);

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  ASSERT_TRUE(AppendFramedResponse(xchn_payload, frame1).ok());
  ASSERT_TRUE(AppendFramedResponse(xchn_payload, frame2).ok());

  timeline.set_max_uncompacted_overview_entries_for_test(64);
  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  BeginIncrementalTraceEvents(std::make_pair(0.0, 8000.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(131072, data_provider, timeline)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));

  const auto& td = timeline.timeline_data();
  int block_parent_idx = -1;
  int block_sibling_idx = -1;
  int split_parent_idx = -1;
  int split_child_idx = -1;
  int split_sibling_idx = -1;
  for (size_t i = 0; i < td.entry_names.size(); ++i) {
    if (td.entry_names[i] == "cross_block_parent") {
      block_parent_idx = static_cast<int>(i);
    } else if (td.entry_names[i] == "block_sibling") {
      block_sibling_idx = static_cast<int>(i);
    } else if (td.entry_names[i] == "split_parent") {
      split_parent_idx = static_cast<int>(i);
    } else if (td.entry_names[i] == "split_child") {
      split_child_idx = static_cast<int>(i);
    } else if (td.entry_names[i] == "split_sibling") {
      split_sibling_idx = static_cast<int>(i);
    }
  }
  ASSERT_GE(block_parent_idx, 0);
  ASSERT_GE(block_sibling_idx, 0);
  EXPECT_EQ(td.entry_levels[block_sibling_idx],
            td.entry_levels[block_parent_idx]);
  EXPECT_DOUBLE_EQ(td.entry_start_times[block_sibling_idx], 10000.2);

  ASSERT_GE(split_parent_idx, 0);
  ASSERT_GE(split_child_idx, 0);
  ASSERT_GE(split_sibling_idx, 0);
  EXPECT_GT(td.entry_levels[split_child_idx],
            td.entry_levels[split_parent_idx]);
  EXPECT_EQ(td.entry_levels[split_sibling_idx],
            td.entry_levels[split_parent_idx]);
  EXPECT_DOUBLE_EQ(td.entry_start_times[split_sibling_idx], 2500.2);
}

TEST(
    TracePbEventParserIncrementalTest,
    IncrementalTraceEventsZoomRefetchPreservesClosingSplitFrameAndSkipsMiddle) {
  constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
  constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;

  // Frame 0: [1000, 1050] ms — while.2682 opens (B, dur=50 ms) + op_f0.
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("while.2682");
  f0.add_interned_strings("op_f0");
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);
  s0->add_deltas(1000000000000ULL);
  s0->add_durations(50000000000ULL | kDurationPhaseBeginBit);
  s0->add_name_refs(1);
  s0->add_event_metadata();
  s0->add_deltas(10000000000ULL);
  s0->add_durations(10000000000ULL);
  s0->add_name_refs(2);
  s0->add_event_metadata();

  // Frame 1: [1200, 1250] ms — intermediate open update (B, dur=250 ms) + op_f1
  xprof::TraceDataResponse f1;
  f1.add_interned_strings("op_f1");
  auto* s1 = f1.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  s1->add_deltas(1000000000000ULL);
  s1->add_durations(250000000000ULL | kDurationPhaseBeginBit);
  s1->add_name_refs(1);
  s1->add_event_metadata();
  s1->add_deltas(210000000000ULL);
  s1->add_durations(10000000000ULL);
  s1->add_name_refs(3);
  s1->add_event_metadata();

  // Frame 2: [1400, 1450] ms — while.2682 closes (E, full dur=450 ms) + op_f2.
  xprof::TraceDataResponse f2;
  f2.add_interned_strings("op_f2");
  auto* s2 = f2.add_complete_events();
  s2->mutable_metadata()->set_process_id(1);
  s2->mutable_metadata()->set_thread_id(10);
  s2->add_deltas(1000000000000ULL);
  s2->add_durations(450000000000ULL | kDurationPhaseEndBit);
  s2->add_name_refs(1);
  s2->add_event_metadata();
  s2->add_deltas(410000000000ULL);
  s2->add_durations(10000000000ULL);
  s2->add_name_refs(4);
  s2->add_event_metadata();

  // Frame 3: [2000, 2050] ms — unrelated later frame.
  xprof::TraceDataResponse f3;
  f3.add_interned_strings("op_f3");
  auto* s3 = f3.add_complete_events();
  s3->mutable_metadata()->set_process_id(1);
  s3->mutable_metadata()->set_thread_id(10);
  s3->add_deltas(2010000000000ULL);
  s3->add_durations(10000000000ULL);
  s3->add_name_refs(5);
  s3->add_event_metadata();

  const absl::StatusOr<std::string> c0 = CompressTraceDataResponse(f0);
  ASSERT_TRUE(c0.ok()) << c0.status();
  const absl::StatusOr<std::string> c1 = CompressTraceDataResponse(f1);
  ASSERT_TRUE(c1.ok()) << c1.status();
  const absl::StatusOr<std::string> c2 = CompressTraceDataResponse(f2);
  ASSERT_TRUE(c2.ok()) << c2.status();
  const absl::StatusOr<std::string> c3 = CompressTraceDataResponse(f3);
  ASSERT_TRUE(c3.ok()) << c3.status();

  ColorPalette palette = ColorPalette::Default();
  IncrementalCacheCleanup cleanup_guard;

  // Verify both the cached-store local replay and spilled compressed-frame
  // refetch preserve Frame 2's closing kDurationEnd full duration while
  // skipping unrelated frames (and Frame 1's B-only frame on spilled replay).
  struct ScopedMaxPackedBytes {
    explicit ScopedMaxPackedBytes(size_t bytes) {
      SetMaxCachedPackedBytesForTest(bytes);
    }
    ~ScopedMaxPackedBytes() {
      SetMaxCachedPackedBytesForTest(512ULL * 1024 * 1024);
    }
  };
  for (const size_t max_cached_bytes : {512ULL * 1024 * 1024, 1ULL}) {
    ScopedMaxPackedBytes scoped_limit(max_cached_bytes);
    const bool expect_cached = (max_cached_bytes > 1ULL);

    DataProvider data_provider;
    Timeline timeline(palette);
    // Force overview compaction so initial load retains the cached block store
    // when within `max_cached_bytes` and spills otherwise.
    timeline.set_max_uncompacted_overview_entries_for_test(2);

    BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
    AppendCompressedTraceEventsIncremental(
        reinterpret_cast<uintptr_t>(c0->data()), c0->size());
    AppendCompressedTraceEventsIncremental(
        reinterpret_cast<uintptr_t>(c1->data()), c1->size());
    AppendCompressedTraceEventsIncremental(
        reinterpret_cast<uintptr_t>(c2->data()), c2->size());
    AppendCompressedTraceEventsIncremental(
        reinterpret_cast<uintptr_t>(c3->data()), c3->size());
    while (StepIncrementalTraceEvents(5000, data_provider, timeline)) {
    }
    EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));
    ASSERT_EQ(HasCachedTraceEventsIncremental(), expect_cached);

    BeginIncrementalTraceEvents(std::make_pair(1005.0, 1045.0), data_provider,
                                timeline);
    if (!expect_cached) {
      EXPECT_TRUE(ShouldEnqueueCompressedFrameIncremental());  // Frame 0 needed
      AppendCompressedTraceEventsIncremental(
          reinterpret_cast<uintptr_t>(c0->data()), c0->size());
      EXPECT_FALSE(
          ShouldEnqueueCompressedFrameIncremental());  // Frame 1 (B-only)
                                                       // skipped
      EXPECT_TRUE(
          ShouldEnqueueCompressedFrameIncremental());  // Frame 2 (E) needed
      AppendCompressedTraceEventsIncremental(
          reinterpret_cast<uintptr_t>(c2->data()), c2->size());
      EXPECT_FALSE(
          ShouldEnqueueCompressedFrameIncremental());  // Frame 3 skipped
    }
    while (StepIncrementalTraceEvents(5000, data_provider, timeline)) {
    }
    EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider, timeline));

    const auto& data = timeline.timeline_data();
    ASSERT_EQ(data.entry_names.size(), 2u);
    EXPECT_EQ(data.entry_names[0], "while.2682");
    EXPECT_DOUBLE_EQ(data.entry_total_times[0], 450000.0);
    EXPECT_EQ(data.entry_names[1], "op_f0");
  }
}

TEST(TracePbEventParserIncrementalTest,
     DispatchResponseDetailsSetsMpmdAndIgnoresOtherKeys) {
  IncrementalCacheCleanup cache_cleanup;
  xprof::TraceDataResponse mpmd_response;
  xprof::DetailItem* full_dma_detail = mpmd_response.add_details();
  full_dma_detail->set_name("full_dma");
  full_dma_detail->set_value(true);
  xprof::DetailItem* single_device_detail = mpmd_response.add_details();
  single_device_detail->set_name("mpmd_single_device_per_stage");
  single_device_detail->set_value(true);
  xprof::DetailItem* mpmd_detail = mpmd_response.add_details();
  mpmd_detail->set_name("mpmd_pipeline_view");
  mpmd_detail->set_value(true);

  const absl::StatusOr<std::string> compressed_mpmd =
      CompressTraceDataResponse(mpmd_response);
  ASSERT_TRUE(compressed_mpmd.ok()) << compressed_mpmd.status();

  ColorPalette palette = ColorPalette::Default();
  Timeline mpmd_timeline(palette);
  DataProvider mpmd_data_provider;
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_mpmd->data()),
      compressed_mpmd->size(), std::nullopt, mpmd_data_provider, mpmd_timeline);
  EXPECT_TRUE(mpmd_timeline.mpmd_pipeline_view_enabled());

  xprof::TraceDataResponse non_mpmd_response;
  xprof::DetailItem* non_mpmd_full_dma = non_mpmd_response.add_details();
  non_mpmd_full_dma->set_name("full_dma");
  non_mpmd_full_dma->set_value(true);

  const absl::StatusOr<std::string> compressed_non_mpmd =
      CompressTraceDataResponse(non_mpmd_response);
  ASSERT_TRUE(compressed_non_mpmd.ok()) << compressed_non_mpmd.status();

  Timeline non_mpmd_timeline(palette);
  DataProvider non_mpmd_data_provider;
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_non_mpmd->data()),
      compressed_non_mpmd->size(), std::nullopt, non_mpmd_data_provider,
      non_mpmd_timeline);
  EXPECT_FALSE(non_mpmd_timeline.mpmd_pipeline_view_enabled());
}

#ifdef __EMSCRIPTEN__
class TracePbEventParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EM_ASM({
      if (typeof global != 'undefined' &&
          typeof global.CustomEvent == 'undefined') {
        global.CustomEvent = function(type, params) {
          this.type = type;
          this.detail = params ? params.detail : null;
        };
      }
      if (typeof window == 'undefined') {
        global.window = {};
        global.window.testResults = {};
        global.window.listeners = {};
        global.window.addEventListener = function(type, listener) {
          global.window.listeners[type] = listener;
        };
        global.window.dispatchEvent = function(event) {
          if (global.window.listeners[event.type]) {
            global.window.listeners[event.type](event);
          }
        };
      } else {
        window.testResults = {};
        window.listeners = {};
        window.addEventListener = function(type, listener) {
          window.listeners[type] = listener;
        };
        window.dispatchEvent = function(event) {
          if (window.listeners[event.type]) {
            window.listeners[event.type](event);
          }
        };
      }
      window.addEventListener(
          'details_received', function(e) {
            window.testResults['details_received'] = {};
            window.testResults['details_received'].received = true;
            window.testResults['details_received'].details = e.detail.details;
          });
    });
  }

  ColorPalette palette_ = ColorPalette::Default();
  Timeline timeline_{palette_};
  DataProvider data_provider_;
};

TEST_F(TracePbEventParserTest, InvalidZstdBuffer) {
  const std::string invalid_zstd = "invalid zstd buffer content";
  const emscripten::val visible_range = emscripten::val::null();
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(invalid_zstd.data()), invalid_zstd.size(),
      visible_range, data_provider_, timeline_);
  EXPECT_EQ(timeline_.data_time_range().duration(), 0);
}

TEST_F(TracePbEventParserTest, InvalidProtobufBuffer) {
  const std::string invalid_proto = "invalid protobuf buffer content";
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(invalid_proto);
  const emscripten::val visible_range = emscripten::val::null();
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), visible_range, data_provider_, timeline_);
  EXPECT_EQ(timeline_.data_time_range().duration(), 0);
}

TEST_F(TracePbEventParserTest, ValidTraceDataWithDetailsAndTimespan) {
  xprof::TraceDataResponse response;
  response.set_full_timespan_start_ps(1000000000);  // 1 ms
  response.set_full_timespan_end_ps(5000000000);    // 5 ms

  auto* detail = response.add_details();
  detail->set_name("full_dma");
  detail->set_value(true);

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  emscripten::val visible_range = emscripten::val::array();
  visible_range.call<void>("push", emscripten::val(2.0));
  visible_range.call<void>("push", emscripten::val(4.0));

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), visible_range, data_provider_, timeline_);

  const emscripten::val results =
      emscripten::val::global("window")["testResults"]["details_received"];
  ASSERT_TRUE(results["received"].as<bool>());
  const emscripten::val details_map = results["details"];
  EXPECT_TRUE(details_map.call<bool>("get", emscripten::val("full_dma")));
}

TEST_F(TracePbEventParserTest, DetailsReceivedDispatchesOnlyFullDma) {
  xprof::TraceDataResponse response;
  xprof::DetailItem* mpmd_detail = response.add_details();
  mpmd_detail->set_name("mpmd_pipeline_view");
  mpmd_detail->set_value(true);
  xprof::DetailItem* single_device_detail = response.add_details();
  single_device_detail->set_name("mpmd_single_device_per_stage");
  single_device_detail->set_value(true);
  xprof::DetailItem* full_dma_detail = response.add_details();
  full_dma_detail->set_name("full_dma");
  full_dma_detail->set_value(false);

  const absl::StatusOr<std::string> compressed_buffer =
      CompressTraceDataResponse(response);
  ASSERT_TRUE(compressed_buffer.ok()) << compressed_buffer.status();

  const emscripten::val visible_range = emscripten::val::undefined();
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer->data()),
      compressed_buffer->size(), visible_range, data_provider_, timeline_);

  EXPECT_TRUE(timeline_.mpmd_pipeline_view_enabled());
  const emscripten::val results =
      emscripten::val::global("window")["testResults"]["details_received"];
  ASSERT_FALSE(results.isUndefined());
  ASSERT_TRUE(results["received"].as<bool>());
  const emscripten::val details_map = results["details"];
  EXPECT_EQ(details_map["size"].as<int>(), 1);
  EXPECT_TRUE(details_map.call<bool>("has", emscripten::val("full_dma")));
  EXPECT_FALSE(details_map.call<bool>("get", emscripten::val("full_dma")));
  EXPECT_FALSE(
      details_map.call<bool>("has", emscripten::val("mpmd_pipeline_view")));
  EXPECT_FALSE(details_map.call<bool>(
      "has", emscripten::val("mpmd_single_device_per_stage")));
}

TEST_F(TracePbEventParserTest,
       InternalOnlyDetailsDoNotDispatchDetailsReceived) {
  xprof::TraceDataResponse response;
  xprof::DetailItem* mpmd_detail = response.add_details();
  mpmd_detail->set_name("mpmd_pipeline_view");
  mpmd_detail->set_value(true);
  xprof::DetailItem* single_device_detail = response.add_details();
  single_device_detail->set_name("mpmd_single_device_per_stage");
  single_device_detail->set_value(true);

  const absl::StatusOr<std::string> compressed_buffer =
      CompressTraceDataResponse(response);
  ASSERT_TRUE(compressed_buffer.ok()) << compressed_buffer.status();

  const emscripten::val visible_range = emscripten::val::undefined();
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer->data()),
      compressed_buffer->size(), visible_range, data_provider_, timeline_);

  EXPECT_TRUE(timeline_.mpmd_pipeline_view_enabled());
  EXPECT_TRUE(
      emscripten::val::global("window")["testResults"]["details_received"]
          .isUndefined());

  // Also verify the incremental XChunk streaming path suppresses host dispatch
  // while still enabling MPMD pipeline view on the timeline.
  Timeline streaming_timeline{palette_};
  DataProvider streaming_data_provider;
  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  AppendFramedBytes(xchn_payload, *compressed_buffer);

  BeginIncrementalTraceEvents(visible_range, streaming_data_provider,
                              streaming_timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());
  while (StepIncrementalTraceEvents(1, streaming_data_provider,
                                    streaming_timeline)) {
  }
  FinishIncrementalTraceEvents(streaming_data_provider, streaming_timeline);

  EXPECT_TRUE(streaming_timeline.mpmd_pipeline_view_enabled());
  EXPECT_TRUE(
      emscripten::val::global("window")["testResults"]["details_received"]
          .isUndefined());
}

TEST_F(TracePbEventParserTest, ValidTraceDataWithInvalidTimespan) {
  xprof::TraceDataResponse response;
  response.set_full_timespan_start_ps(5000000000);  // 5 ms
  response.set_full_timespan_end_ps(1000000000);    // 1 ms (start > end)

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  const emscripten::val visible_range = emscripten::val::undefined();

  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), visible_range, data_provider_, timeline_);
}

TEST_F(TracePbEventParserTest, IncrementalTraceEventsXchnMultiFrame) {
  auto append_framed = [](std::string& out,
                          const xprof::TraceDataResponse& frame) {
    std::string serialized;
    ASSERT_TRUE(frame.SerializeToString(&serialized));
    const std::string compressed =
        *tensorflow::profiler::ZstdCompression::Compress(serialized);
    const uint32_t len = static_cast<uint32_t>(compressed.size());
    const char len_bytes[4] = {
        static_cast<char>(len & 0xff),
        static_cast<char>((len >> 8) & 0xff),
        static_cast<char>((len >> 16) & 0xff),
        static_cast<char>((len >> 24) & 0xff),
    };
    out.append(len_bytes, 4);
    out.append(compressed);
  };

  xprof::TraceDataResponse frame1;
  frame1.set_full_timespan_start_ps(1000000000);
  frame1.set_full_timespan_end_ps(5000000000);
  frame1.add_interned_strings("");
  frame1.add_interned_strings("op_a");
  auto* proc1 = frame1.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops");
  auto* series1 = frame1.add_complete_events();
  series1->mutable_metadata()->set_process_id(1);
  series1->mutable_metadata()->set_thread_id(10);
  series1->add_deltas(1000000000);
  series1->add_durations(500000000);
  series1->add_name_refs(1);
  series1->add_event_metadata();

  xprof::TraceDataResponse frame2;
  frame2.add_interned_strings("op_b");
  auto* proc2 = frame2.mutable_metadata()->add_processes();
  proc2->set_id(2);
  proc2->set_name("host:0 /device:TPU:1");
  auto* th2 = proc2->add_threads();
  th2->set_id(20);
  th2->set_name("XLA Ops");
  auto* series2 = frame2.add_complete_events();
  series2->mutable_metadata()->set_process_id(2);
  series2->mutable_metadata()->set_thread_id(20);
  series2->add_deltas(2000000000);
  series2->add_durations(500000000);
  series2->add_name_refs(2);
  series2->add_event_metadata();

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  append_framed(xchn_payload, frame1);
  append_framed(xchn_payload, frame2);
  // Trailing frame length 0xFFFFFFF0 exercises the 32-bit size_t overflow guard
  // (frame_len > data_size - offset).
  xchn_payload.append("\xF0\xFF\xFF\xFF", 4);

  const emscripten::val visible_range = emscripten::val::undefined();
  BeginIncrementalTraceEvents(visible_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());

  EXPECT_TRUE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op_a");

  EXPECT_FALSE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "op_b");

  FinishIncrementalTraceEvents(data_provider_, timeline_);
  EXPECT_EQ(timeline_.data_time_range().duration(), 4000.0);
  EXPECT_TRUE(timeline_.full_trace_loaded());
}

TEST_F(TracePbEventParserTest,
       IncrementalTraceEventsXchnOpenSpanExtensionAcrossFrames) {
  constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
  constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;

  auto append_framed = [](std::string& out,
                          const xprof::TraceDataResponse& frame) {
    std::string serialized;
    ASSERT_TRUE(frame.SerializeToString(&serialized));
    const std::string compressed =
        *tensorflow::profiler::ZstdCompression::Compress(serialized);
    const uint32_t len = static_cast<uint32_t>(compressed.size());
    const char len_bytes[4] = {
        static_cast<char>(len & 0xff),
        static_cast<char>((len >> 8) & 0xff),
        static_cast<char>((len >> 16) & 0xff),
        static_cast<char>((len >> 24) & 0xff),
    };
    out.append(len_bytes, 4);
    out.append(compressed);
  };

  // Frame 1: while_loop starts at 1000 us with partial duration 500 us (tagged
  // with kDurationPhaseBeginBit), and child_1 runs [1100, 1300] us (200 us).
  xprof::TraceDataResponse frame1;
  frame1.add_interned_strings("");
  frame1.add_interned_strings("while_loop");
  frame1.add_interned_strings("child_1");
  auto* proc1 = frame1.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops");
  auto* series1 = frame1.add_complete_events();
  series1->mutable_metadata()->set_process_id(1);
  series1->mutable_metadata()->set_thread_id(10);
  series1->add_deltas(1000000000ULL);
  series1->add_durations(500000000ULL | kDurationPhaseBeginBit);
  series1->add_name_refs(1);
  series1->add_event_metadata();
  series1->add_deltas(100000000ULL);
  series1->add_durations(200000000ULL);
  series1->add_name_refs(2);
  series1->add_event_metadata();

  // Frame 2: while_loop closes at 2500 us with full duration 1500 us (tagged
  // with kDurationPhaseEndBit), and child_2 runs [1600, 1900] us (300 us).
  xprof::TraceDataResponse frame2;
  frame2.add_interned_strings("child_2");
  auto* series2 = frame2.add_complete_events();
  series2->mutable_metadata()->set_process_id(1);
  series2->mutable_metadata()->set_thread_id(10);
  series2->add_deltas(1000000000ULL);
  series2->add_durations(1500000000ULL | kDurationPhaseEndBit);
  series2->add_name_refs(1);
  series2->add_event_metadata();
  series2->add_deltas(600000000ULL);
  series2->add_durations(300000000ULL);
  series2->add_name_refs(3);
  series2->add_event_metadata();

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  append_framed(xchn_payload, frame1);
  append_framed(xchn_payload, frame2);
  // Trailing frame length 0xFFFFFFF0 exercises the 32-bit size_t overflow guard
  // (frame_len > data_size - offset).
  xchn_payload.append("\xF0\xFF\xFF\xFF", 4);

  const emscripten::val visible_range = emscripten::val::undefined();
  BeginIncrementalTraceEvents(visible_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());

  EXPECT_TRUE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_DOUBLE_EQ(timeline_.timeline_data().entry_total_times[0], 500.0);
  EXPECT_DOUBLE_EQ(timeline_.timeline_data().entry_self_times[0], 300.0);

  EXPECT_FALSE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  FinishIncrementalTraceEvents(data_provider_, timeline_);

  const auto& data = timeline_.timeline_data();
  ASSERT_EQ(data.entry_names.size(), 3u);
  EXPECT_EQ(data.entry_names[0], "while_loop");
  EXPECT_DOUBLE_EQ(data.entry_start_times[0], 1000.0);
  EXPECT_DOUBLE_EQ(data.entry_total_times[0], 1500.0);
  EXPECT_DOUBLE_EQ(data.entry_self_times[0], 1000.0);
  EXPECT_EQ(data.entry_levels[0], 0);
  EXPECT_EQ(data.entry_levels[1], 1);
  EXPECT_EQ(data.entry_levels[2], 1);
}

TEST_F(TracePbEventParserTest,
       IncrementalTraceEventsZoomRefetchFiltersEventsToWindow) {
  xprof::TraceDataResponse frame;
  frame.add_interned_strings("");
  frame.add_interned_strings("op_early");
  frame.add_interned_strings("op_middle");
  frame.add_interned_strings("op_late");
  auto* proc = frame.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* series = frame.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);
  // op_early: [1000, 1200] us
  series->add_deltas(1000000000ULL);
  series->add_durations(200000000ULL);
  series->add_name_refs(1);
  series->add_event_metadata();
  // op_middle: [2000, 2200] us
  series->add_deltas(1000000000ULL);
  series->add_durations(200000000ULL);
  series->add_name_refs(2);
  series->add_event_metadata();
  // op_late: [3000, 3200] us
  series->add_deltas(1000000000ULL);
  series->add_durations(200000000ULL);
  series->add_name_refs(3);
  series->add_event_metadata();

  std::string serialized;
  ASSERT_TRUE(frame.SerializeToString(&serialized));
  const std::string compressed =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  // Initial load: all 3 events loaded, full_trace_loaded == true.
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed.data()), compressed.size());
  EXPECT_TRUE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 3u);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().end(), 3200.0);

  // Zoom refetch for [1.8, 2.4] ms ([1800, 2400] us): only op_middle is kept,
  // while data_time_range preserves [1000, 3200] us.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 1.8);
  zoom_range.set(1, 2.4);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(compressed.data()), compressed.size());
  // Intermediate step during refetch must not replace existing timeline_data
  // until FinishIncrementalTraceEvents completes.
  EXPECT_FALSE(StepIncrementalTraceEvents(1, data_provider_, timeline_));
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 3u);

  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op_middle");
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.data_time_range().end(), 3200.0);
  EXPECT_TRUE(timeline_.window_at_full_resolution());
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().start(), 1800.0);
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().end(), 2400.0);

  // Zooming back out restores the stashed full-trace overview in O(1) without
  // re-decompressing or re-parsing frames.
  EXPECT_TRUE(RestoreFullTraceOverviewIfAvailable(timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 3u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op_early");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "op_middle");
  EXPECT_EQ(timeline_.timeline_data().entry_names[2], "op_late");
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().start(), 1000.0);
  EXPECT_DOUBLE_EQ(timeline_.last_fetch_request_range().end(), 3200.0);
}

TEST_F(TracePbEventParserTest,
       IncrementalTraceEventsZoomRefetchSkipsNonOverlappingFrames) {
  auto compress_frame = [](const xprof::TraceDataResponse& frame) {
    std::string serialized;
    EXPECT_TRUE(frame.SerializeToString(&serialized));
    return *tensorflow::profiler::ZstdCompression::Compress(serialized);
  };

  // Frame 0: [100, 200] ms (100,000..200,000 us), interned strings ["", "op0"]
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("op0");
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);
  s0->add_deltas(100000000000ULL);
  s0->add_durations(100000000000ULL);
  s0->add_name_refs(1);
  s0->add_event_metadata();

  // Frame 1: [2000, 2100] ms, cumulative interned string ["op1"] (id 2)
  xprof::TraceDataResponse f1;
  f1.add_interned_strings("op1");
  auto* s1 = f1.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  s1->add_deltas(2000000000000ULL);
  s1->add_durations(100000000000ULL);
  s1->add_name_refs(2);
  s1->add_event_metadata();

  // Frame 2: [5000, 5100] ms, cumulative interned string ["op2"] (id 3)
  xprof::TraceDataResponse f2;
  f2.add_interned_strings("op2");
  auto* s2 = f2.add_complete_events();
  s2->mutable_metadata()->set_process_id(1);
  s2->mutable_metadata()->set_thread_id(10);
  s2->add_deltas(5000000000000ULL);
  s2->add_durations(100000000000ULL);
  s2->add_name_refs(3);
  s2->add_event_metadata();

  const std::string c0 = compress_frame(f0);
  const std::string c1 = compress_frame(f1);
  const std::string c2 = compress_frame(f2);

  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c1.data()),
                                         c1.size());
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c2.data()),
                                         c2.size());
  EXPECT_TRUE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 3u);

  // Zoom refetch for [1900, 2200] ms: Frame 0 ([100, 200] ms) and Frame 2
  // ([5000, 5100] ms) are outside the 500 ms margin around [1900, 2200] ms, so
  // ShouldEnqueueCompressedFrameIncremental() returns false for Frames 0 and 2
  // and true for Frame 1.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 1900.0);
  zoom_range.set(1, 2200.0);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  EXPECT_FALSE(ShouldEnqueueCompressedFrameIncremental());  // Frame 0 skipped
  EXPECT_TRUE(ShouldEnqueueCompressedFrameIncremental());   // Frame 1 needed
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c1.data()),
                                         c1.size());
  EXPECT_FALSE(ShouldEnqueueCompressedFrameIncremental());  // Frame 2 skipped
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));

  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "op1");
}

TEST_F(
    TracePbEventParserTest,
    InitialOverviewLoadCompactsLargeTraceAndSubWindowPreservesFullResolution) {
  xprof::TraceDataResponse response;
  response.add_interned_strings("");
  response.add_interned_strings("op");
  response.set_full_timespan_start_ps(0);
  response.set_full_timespan_end_ps(5000000000ULL);  // 5 ms
  auto* process = response.mutable_metadata()->add_processes();
  process->set_id(1);
  process->set_name("Host");
  auto* thread = process->add_threads();
  thread->set_id(1);
  thread->set_name("Thread 1");

  auto* series = response.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(1);
  constexpr int kNumEvents = 500;
  for (int i = 0; i < kNumEvents; ++i) {
    series->add_deltas(10000000ULL);    // +10 us each
    series->add_durations(5000000ULL);  // 5 us duration
    series->add_name_refs(1);
    series->add_event_metadata();
  }

  std::string serialized_proto;
  ASSERT_TRUE(response.SerializeToString(&serialized_proto));
  const std::string compressed_buffer =
      *tensorflow::profiler::ZstdCompression::Compress(serialized_proto);

  timeline_.set_max_uncompacted_overview_entries_for_test(64);
  timeline_.set_incremental_fetch_enabled(true);

  // 1. Initial overview load (no visible_range_from_url) configures
  // is_compacted(true) and compacts the 500 events down to MIP dominant events.
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), emscripten::val::undefined(), data_provider_,
      timeline_);
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_FALSE(timeline_.window_at_full_resolution());
  EXPECT_FALSE(timeline_.full_trace_loaded());
  EXPECT_LT(timeline_.timeline_data().entry_start_times.size(), 300u);

  // 2. Subsequent sub-window fetch (with visible_range_from_url) configures
  // is_compacted(false) and preserves all 500 events at full resolution.
  emscripten::val fetch_range_ms = emscripten::val::array();
  fetch_range_ms.call<void>("push", emscripten::val(0.0));
  fetch_range_ms.call<void>("push", emscripten::val(5.0));
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(compressed_buffer.data()),
      compressed_buffer.size(), fetch_range_ms, data_provider_, timeline_);
  EXPECT_FALSE(timeline_.is_compacted());
  EXPECT_TRUE(timeline_.window_at_full_resolution());
  EXPECT_EQ(timeline_.timeline_data().entry_start_times.size(), 500u);
}

TEST_F(TracePbEventParserTest,
       CompactsFullOverviewToMipPyramidsAndRestoresOnZoomOut) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("sc_op");
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0 SparseCore:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("SparseCore Ops");
  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);

  // Add 2000 closely-packed back-to-back events on the same level so that
  // BuildLevelMipPyramids merges multiple events into MIP tiles and
  // CompactTimelineDataToMipOverview compacts raw entries down to dominant
  // tile events.
  constexpr int kNumEvents = 2000;
  for (int i = 0; i < kNumEvents; ++i) {
    s0->add_deltas(i == 0 ? 1000000ULL : 1000000ULL);
    s0->add_durations(900000ULL);
    s0->add_name_refs(1);
    s0->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_TRUE(HasCachedTraceEventsIncremental());
  EXPECT_TRUE(timeline_.is_showing_full_overview());

  const size_t compacted_overview_entries =
      timeline_.timeline_data().entry_start_times.size();
  EXPECT_GT(compacted_overview_entries, 0u);
  EXPECT_LT(compacted_overview_entries, static_cast<size_t>(kNumEvents));
  ASSERT_FALSE(timeline_.level_mip_pyramids_for_test().empty());
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].has_pyramid);
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].raw_events_compacted);

  // Zoom in to a narrow window [0.0, 0.5] ms (~500 events, <= 512 so raw
  // uncompacted events are retained in the zoomed-in window). Because
  // HasCachedTraceEventsIncremental() is true, no compressed frame needs to be
  // re-enqueued or re-decompressed.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 0.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.HasSavedFullOverview());
  EXPECT_EQ(timeline_.timeline_data().entry_start_times.size(), 500u);

  // Zoom out and restore the mip-compacted full overview from memory.
  EXPECT_TRUE(RestoreFullTraceOverviewIfAvailable(timeline_));
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_EQ(timeline_.timeline_data().entry_start_times.size(),
            compacted_overview_entries);
  ASSERT_FALSE(timeline_.level_mip_pyramids_for_test().empty());
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].has_pyramid);
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test()[0].raw_events_compacted);
}

TEST_F(TracePbEventParserTest,
       FiltersRefetchByExpandedTracksAndBorrowsOverviewPyramids) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("tc_op");
  f0.add_interned_strings("sc_op");

  // Process 1 (TensorCore, expanded by default) and Process 2 (SparseCore,
  // collapsed by default because its name contains "SparseCore").
  auto* proc1 = f0.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops");

  auto* proc2 = f0.mutable_metadata()->add_processes();
  proc2->set_id(2);
  proc2->set_name("host:0 /device:TPU:0 SparseCore:0");
  auto* th2 = proc2->add_threads();
  th2->set_id(20);
  th2->set_name("SparseCore Ops");

  auto* s1 = f0.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 100; ++i) {
    s1->add_deltas(10000000ULL);
    s1->add_durations(9000000ULL);
    s1->add_name_refs(1);
    s1->add_event_metadata();
  }

  auto* s2 = f0.add_complete_events();
  s2->mutable_metadata()->set_process_id(2);
  s2->mutable_metadata()->set_thread_id(20);
  for (int i = 0; i < 1000; ++i) {
    s2->add_deltas(1000000ULL);
    s2->add_durations(900000ULL);
    s2->add_name_refs(2);
    s2->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));

  // On zoom refetch into [0.0, 0.5] ms, only Process 1 (expanded) is in
  // ComputeActiveTracksForFetch(); Process 2 (SparseCore, collapsed) is
  // skipped, so only Process 1's 50 events are decoded into entry_names, while
  // Process 2's level borrows its overview MIP pyramid from
  // saved_full_overview_.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 0.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));

  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 50u);
  for (const std::string& name : timeline_.timeline_data().entry_names) {
    EXPECT_EQ(name, "tc_op");
  }
  ASSERT_GE(timeline_.level_mip_pyramids_for_test().size(), 2u);
  // Process 2's level borrowed its MIP pyramid from saved_full_overview_.
  EXPECT_TRUE(timeline_.level_mip_pyramids_for_test().back().has_pyramid);
  EXPECT_FALSE(
      timeline_.level_mip_pyramids_for_test().back().levels[1].tiles.empty());
  EXPECT_FALSE(timeline_.HasUnloadedVisibleExpandedTracks());
}

TEST_F(TracePbEventParserTest,
       ZoomInSynchronousRefetchThenZoomOutRestoresFullOverview) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("early_op");
  f0.add_interned_strings("late_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000000ULL);  // 10,000 ms

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* s1 = f0.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);

  // Add 600 events in [0 ms, 1200 ms] ("early_op") and 600 events in
  // [8850 ms, 10000 ms] ("late_op") across a 10,000 ms trace.
  for (int i = 0; i < 600; ++i) {
    s1->add_deltas(i == 0 ? 1000000000ULL : 1000000000ULL);  // +1 ms each
    s1->add_durations(800000000ULL);                         // 0.8 ms
    s1->add_name_refs(1);
    s1->add_event_metadata();
  }
  for (int i = 0; i < 600; ++i) {
    // Jump from 600 ms to 8850 ms on the first late event (+8250 ms).
    s1->add_deltas(i == 0 ? 8250000000000ULL : 1000000000ULL);
    s1->add_durations(800000000ULL);
    s1->add_name_refs(2);
    s1->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(512);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_FALSE(timeline_.is_incremental_loading());
  const size_t full_overview_count =
      timeline_.timeline_data().entry_names.size();
  EXPECT_GT(full_overview_count, 0u);

  // Wire kFetchData to execute the incremental cached refetch synchronously,
  // reproducing the exact call order when stepIncrementalTraceEvents finishes
  // in a single synchronous step.
  int fetch_count = 0;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type != kFetchData) return;
        ++fetch_count;
        const double start_ms =
            std::any_cast<double>(detail.at(std::string(kFetchDataStart)));
        const double end_ms =
            std::any_cast<double>(detail.at(std::string(kFetchDataEnd)));
        emscripten::val range = emscripten::val::array();
        range.set(0, start_ms);
        range.set(1, end_ms);
        BeginIncrementalTraceEvents(range, data_provider_, timeline_);
        while (StepIncrementalTraceEvents(65536, data_provider_, timeline_)) {
        }
        FinishIncrementalTraceEvents(data_provider_, timeline_);
      });

  // 1. Zoom into [9000 ms, 9500 ms] (500 ms window -> 1500 ms fetch window
  // [8500 ms, 10000 ms]).
  timeline_.SetVisibleRange({9000000.0, 9500000.0});
  timeline_.MaybeRequestData();
  EXPECT_EQ(fetch_count, 1);
  // Regression check: synchronous completion inside kFetchData callback must
  // NOT leave is_incremental_loading() stuck at true.
  EXPECT_FALSE(timeline_.is_incremental_loading());
  EXPECT_FALSE(timeline_.is_showing_full_overview());
  EXPECT_TRUE(timeline_.HasSavedFullOverview());
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 600u);
  for (const std::string& name : timeline_.timeline_data().entry_names) {
    EXPECT_EQ(name, "late_op");
  }

  // 2. Zoom out to [550 ms, 9950 ms] (9400 ms visible window, 94% of the
  // 10,000 ms trace) and call MaybeRequestData(). This must immediately restore
  // the full overview including all "early_op" and "late_op" events.
  timeline_.SetVisibleRange({550000.0, 9950000.0});
  timeline_.MaybeRequestData();
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_FALSE(timeline_.is_incremental_loading());
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), full_overview_count);
  bool has_early_op = false;
  bool has_late_op = false;
  for (const std::string& name : timeline_.timeline_data().entry_names) {
    if (name == "early_op") has_early_op = true;
    if (name == "late_op") has_late_op = true;
  }
  EXPECT_TRUE(has_early_op);
  EXPECT_TRUE(has_late_op);

  // 3. Also verify that if a filtered refetch is in flight when the user zooms
  // out and restores the full overview, finishing the stale filtered refetch
  // does not overwrite the restored full overview.
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 8850.0);
  zoom_range.set(1, 10000.0);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  StepIncrementalTraceEvents(100, data_provider_, timeline_);
  // User zooms out to [550 ms, 9950 ms] before FinishIncrementalTraceEvents:
  timeline_.SetVisibleRange({550000.0, 9950000.0});
  timeline_.MaybeRequestData();
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  // Stale filtered refetch finishes late:
  FinishIncrementalTraceEvents(data_provider_, timeline_);
  EXPECT_TRUE(timeline_.is_showing_full_overview());
  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), full_overview_count);
}

TEST_F(TracePbEventParserTest,
       ZoomRefetchWhenAllVisibleProcessesCollapsedDoesNotDropAllEvents) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("sc_op");
  // SparseCore processes default to collapsed, so ComputeActiveTracksForFetch
  // returns an empty set when no other process is expanded.
  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(2);
  proc->set_name("host:0 /device:TPU:0 SparseCore:0");
  auto* th = proc->add_threads();
  th->set_id(20);
  th->set_name("SparseCore Ops");

  auto* s0 = f0.add_complete_events();
  s0->mutable_metadata()->set_process_id(2);
  s0->mutable_metadata()->set_thread_id(20);
  for (int i = 0; i < 100; ++i) {
    s0->add_deltas(10000000ULL);
    s0->add_durations(9000000ULL);
    s0->add_name_refs(1);
    s0->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(32);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  FinishIncrementalTraceEvents(data_provider_, timeline_);
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 0.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  FinishIncrementalTraceEvents(data_provider_, timeline_);

  EXPECT_EQ(timeline_.timeline_data().entry_names.size(), 50u);
  EXPECT_FALSE(timeline_.HasUnloadedVisibleExpandedTracks());
}

TEST_F(TracePbEventParserTest,
       OverviewPreCompactionPreservedWhenExceedingMaxCachedPackedBytes) {
  struct ScopedMaxPackedBytes {
    explicit ScopedMaxPackedBytes(size_t bytes) {
      SetMaxCachedPackedBytesForTest(bytes);
    }
    ~ScopedMaxPackedBytes() {
      SetMaxCachedPackedBytesForTest(512ULL * 1024 * 1024);
    }
  } scoped_limit(60000);

  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("dense_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(10000000000ULL);

  auto* proc1 = f0.mutable_metadata()->add_processes();
  proc1->set_id(1);
  proc1->set_name("host:0 /device:TPU:0");
  auto* th1 = proc1->add_threads();
  th1->set_id(10);
  th1->set_name("XLA Ops 1");

  auto* proc2 = f0.mutable_metadata()->add_processes();
  proc2->set_id(2);
  proc2->set_name("host:0 /device:TPU:1");
  auto* th2 = proc2->add_threads();
  th2->set_id(20);
  th2->set_name("XLA Ops 2");

  // Process 1 has 9000 sub-quantum events (> 8192, ~54 KB packed) so
  // ProcessOverviewSampledSeriesSlice pre-compacts them and sets
  // overview_pre_compacted = true, while Process 2's packed blocks push
  // building_packed_bytes past the 60 KB test limit and trigger
  // ResetBuildingTraceStore().
  auto* s1 = f0.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 9000; ++i) {
    s1->add_deltas(1000ULL);
    s1->add_durations(500ULL);
    s1->add_name_refs(1);
    s1->add_event_metadata();
  }

  auto* s2 = f0.add_complete_events();
  s2->mutable_metadata()->set_process_id(2);
  s2->mutable_metadata()->set_thread_id(20);
  for (int i = 0; i < 2000; ++i) {
    s2->add_deltas(1000000ULL + static_cast<uint64_t>(i) * 1000ULL);
    s2->add_durations(500000ULL + static_cast<uint64_t>(i) * 100ULL);
    s2->add_name_refs(1);
    auto* meta = s2->add_event_metadata();
    meta->set_serial(static_cast<uint64_t>(i + 1) * 1000ULL);
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  // Set a high uncompacted entry threshold so surviving post-sampled events are
  // well below max_uncompacted_overview_entries().
  timeline_.set_max_uncompacted_overview_entries_for_test(100000);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  while (StepIncrementalTraceEvents(5000, data_provider_, timeline_)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
  EXPECT_TRUE(timeline_.is_compacted());
  EXPECT_FALSE(timeline_.full_trace_loaded());
  EXPECT_FALSE(timeline_.window_at_full_resolution());
}

TEST_F(TracePbEventParserTest,
       InitialUrlVisibleRangeFullSessionLoadArmsCachedTraceStore) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("url_window_op");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(100000000000ULL);

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  auto* series = f0.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 64; ++i) {
    series->add_deltas(1000000ULL);
    series->add_durations(500000ULL);
    series->add_name_refs(1);
    series->add_event_metadata();
  }

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(16);
  emscripten::val url_range = emscripten::val::array();
  url_range.set(0, 10.0);
  url_range.set(1, 20.0);
  BeginIncrementalTraceEvents(url_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  while (StepIncrementalTraceEvents(100, data_provider_, timeline_)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_TRUE(HasCachedTraceEventsIncremental());
}

TEST_F(TracePbEventParserTest,
       CachedRefetchFiltersAsyncFlowAndCounterEventsToWindow) {
  xprof::TraceDataResponse f0;
  f0.add_interned_strings("");
  f0.add_interned_strings("sync_op");
  f0.add_interned_strings("async_op");
  f0.add_interned_strings("mem_bw");
  f0.set_full_timespan_start_ps(0ULL);
  f0.set_full_timespan_end_ps(100000000000ULL);

  auto* proc = f0.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");

  // In-window complete events at 1.0 ms and 2.0 ms linked by sync flow_id=777,
  // plus enough events to exceed max_uncompacted_overview_entries (16) so the
  // initial load caches the block store.
  auto* sync_series = f0.add_complete_events();
  sync_series->mutable_metadata()->set_process_id(1);
  sync_series->mutable_metadata()->set_thread_id(10);
  for (int i = 0; i < 40; ++i) {
    sync_series->add_deltas(100000ULL);
    sync_series->add_durations(50000ULL);
    sync_series->add_name_refs(1);
    auto* meta = sync_series->add_event_metadata();
    if (i == 10 || i == 20) {
      meta->set_flow_id(777);
    }
  }

  // >16384 out-of-window async flow events starting at 50 ms (50,000,000,000
  // ps). Without filtering cached_async_flow_events on a zoomed refetch, these
  // saturate kMaxIncrementalFlowEvents (16384) and drop sync flow_id=777.
  auto* async_series = f0.add_async_events();
  async_series->mutable_metadata()->set_process_id(1);
  async_series->mutable_metadata()->set_name_ref(2);
  for (int i = 0; i < 16500; ++i) {
    async_series->add_deltas(i == 0 ? 50000000000ULL : 1000ULL);
    async_series->add_durations(500ULL);
    auto* meta = async_series->add_event_metadata();
    meta->set_flow_id(static_cast<uint64_t>(1000 + i / 2));
  }

  // Counter series with in-window points at 1 ms / 2 ms (values 10.0, 20.0) and
  // an out-of-window spike at 90 ms (value 999.0).
  auto* counter_series = f0.add_counter_events();
  counter_series->mutable_metadata()->set_process_id(1);
  counter_series->mutable_metadata()->set_name_ref(3);
  counter_series->add_deltas(1000000000ULL);
  counter_series->add_event_metadata()->set_counter_value_double(10.0);
  counter_series->add_deltas(1000000000ULL);
  counter_series->add_event_metadata()->set_counter_value_double(20.0);
  counter_series->add_deltas(88000000000ULL);
  counter_series->add_event_metadata()->set_counter_value_double(999.0);

  std::string serialized;
  ASSERT_TRUE(f0.SerializeToString(&serialized));
  const std::string c0 =
      *tensorflow::profiler::ZstdCompression::Compress(serialized);

  timeline_.set_max_uncompacted_overview_entries_for_test(16);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(reinterpret_cast<uintptr_t>(c0.data()),
                                         c0.size());
  while (StepIncrementalTraceEvents(20000, data_provider_, timeline_)) {
  }
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_TRUE(HasCachedTraceEventsIncremental());

  // Zoomed cached refetch over [0.0, 5.0] ms ([0, 5000] us).
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 0.0);
  zoom_range.set(1, 5.0);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  while (StepIncrementalTraceEvents(5000, data_provider_, timeline_)) {
  }
  FinishIncrementalTraceEvents(data_provider_, timeline_);

  EXPECT_TRUE(
      timeline_.timeline_data().flow_lines_by_flow_id.contains("777"));
  EXPECT_LE(timeline_.fetched_data_time_range().end(), 5000.0);
  bool found_counter = false;
  for (const auto& group : timeline_.timeline_data().groups) {
    if (group.type == Group::Type::kCounter && group.name == "mem_bw") {
      found_counter = true;
      const auto& counters =
          timeline_.timeline_data().counter_data_by_group_index;
      const auto it = counters.find(
          static_cast<int>(&group - timeline_.timeline_data().groups.data()));
      ASSERT_NE(it, counters.end());
      EXPECT_EQ(it->second.values.size(), 2);
      EXPECT_DOUBLE_EQ(it->second.min_value, 10.0);
      EXPECT_DOUBLE_EQ(it->second.max_value, 999.0);
    }
  }
  EXPECT_TRUE(found_counter);
}

TEST_F(TracePbEventParserTest,
       NetworkRefetchesWithWindowSpecificStringTablesPreserveEventNames) {
  auto build_window_response = [](absl::string_view name_1,
                                  absl::string_view name_2, uint64_t ts1_ps,
                                  uint64_t ts2_ps) {
    xprof::TraceDataResponse resp;
    resp.set_full_timespan_start_ps(0ULL);
    resp.set_full_timespan_end_ps(10000000000ULL);  // 10 ms
    resp.add_interned_strings("");
    resp.add_interned_strings(std::string(name_1));
    resp.add_interned_strings(std::string(name_2));
    auto* proc = resp.mutable_metadata()->add_processes();
    proc->set_id(1);
    proc->set_name("host:0 /device:TPU:0");
    auto* th = proc->add_threads();
    th->set_id(10);
    th->set_name("XLA Ops");
    auto* series = resp.add_complete_events();
    series->mutable_metadata()->set_process_id(1);
    series->mutable_metadata()->set_thread_id(10);
    series->add_deltas(ts1_ps);
    series->add_durations(100000000ULL);
    series->add_name_refs(1);
    series->add_event_metadata();
    series->add_deltas(ts2_ps - ts1_ps);
    series->add_durations(100000000ULL);
    series->add_name_refs(2);
    series->add_event_metadata();
    std::string serialized;
    EXPECT_TRUE(resp.SerializeToString(&serialized));
    return *tensorflow::profiler::ZstdCompression::Compress(serialized);
  };

  // 1. Initial load via ParseAndProcessCompressedTraceEvents (legacy overview
  // load when resolution > 0).
  const std::string overview_compressed = build_window_response(
      "overview_op_1", "overview_op_2", 1000000000ULL, 5000000000ULL);
  ParseAndProcessCompressedTraceEvents(
      reinterpret_cast<uintptr_t>(overview_compressed.data()),
      overview_compressed.size(), emscripten::val::undefined(), data_provider_,
      timeline_);
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "overview_op_1");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "overview_op_2");

  // 2. First zoom-in network refetch for [1.0, 3.0] ms with its own
  // window-specific interned_strings table.
  const std::string zoom1_compressed = build_window_response(
      "zoom1_op_a", "zoom1_op_b", 1200000000ULL, 2200000000ULL);
  emscripten::val zoom1_range = emscripten::val::array();
  zoom1_range.set(0, 1.0);
  zoom1_range.set(1, 3.0);
  BeginIncrementalTraceEvents(zoom1_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(zoom1_compressed.data()),
      zoom1_compressed.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "zoom1_op_a");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "zoom1_op_b");

  // 3. Second zoom-out / pan network refetch for [0.5, 6.0] ms with a fresh
  // window-specific interned_strings table where indices 1 and 2 map to
  // completely different event names.
  const std::string zoom2_compressed = build_window_response(
      "zoom2_op_x", "zoom2_op_y", 800000000ULL, 4500000000ULL);
  emscripten::val zoom2_range = emscripten::val::array();
  zoom2_range.set(0, 0.5);
  zoom2_range.set(1, 6.0);
  BeginIncrementalTraceEvents(zoom2_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(zoom2_compressed.data()),
      zoom2_compressed.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 2u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "zoom2_op_x");
  EXPECT_EQ(timeline_.timeline_data().entry_names[1], "zoom2_op_y");
}

TEST_F(TracePbEventParserTest,
       NetworkRefetchAfterUncachedIncrementalLoadUsesResponseStrings) {
  auto compress_frame = [](const xprof::TraceDataResponse& frame) {
    std::string serialized;
    EXPECT_TRUE(frame.SerializeToString(&serialized));
    return *tensorflow::profiler::ZstdCompression::Compress(serialized);
  };

  xprof::TraceDataResponse init_resp;
  init_resp.set_full_timespan_start_ps(0ULL);
  init_resp.set_full_timespan_end_ps(10000000000ULL);
  init_resp.add_interned_strings("");
  init_resp.add_interned_strings("initial_op");
  auto* proc = init_resp.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("host:0 /device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(10);
  th->set_name("XLA Ops");
  auto* s0 = init_resp.add_complete_events();
  s0->mutable_metadata()->set_process_id(1);
  s0->mutable_metadata()->set_thread_id(10);
  s0->add_deltas(1000000000ULL);
  s0->add_durations(200000000ULL);
  s0->add_name_refs(1);
  s0->add_event_metadata();

  const std::string c_init = compress_frame(init_resp);
  BeginIncrementalTraceEvents(emscripten::val::undefined(), data_provider_,
                              timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c_init.data()), c_init.size());
  EXPECT_TRUE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  EXPECT_FALSE(HasCachedTraceEventsIncremental());
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "initial_op");

  // Network refetch response with its own interned_strings table where index 1
  // is "refetch_window_op" (enqueued directly without calling
  // ShouldEnqueueCompressedFrameIncremental()).
  xprof::TraceDataResponse refetch_resp;
  refetch_resp.set_full_timespan_start_ps(0ULL);
  refetch_resp.set_full_timespan_end_ps(10000000000ULL);
  refetch_resp.add_interned_strings("");
  refetch_resp.add_interned_strings("refetch_window_op");
  auto* r_proc = refetch_resp.mutable_metadata()->add_processes();
  r_proc->set_id(1);
  r_proc->set_name("host:0 /device:TPU:0");
  auto* r_th = r_proc->add_threads();
  r_th->set_id(10);
  r_th->set_name("XLA Ops");
  auto* s1 = refetch_resp.add_complete_events();
  s1->mutable_metadata()->set_process_id(1);
  s1->mutable_metadata()->set_thread_id(10);
  s1->add_deltas(2000000000ULL);
  s1->add_durations(200000000ULL);
  s1->add_name_refs(1);
  s1->add_event_metadata();

  const std::string c_refetch = compress_frame(refetch_resp);
  emscripten::val zoom_range = emscripten::val::array();
  zoom_range.set(0, 1.5);
  zoom_range.set(1, 2.5);
  BeginIncrementalTraceEvents(zoom_range, data_provider_, timeline_);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(c_refetch.data()), c_refetch.size());
  EXPECT_FALSE(FinishIncrementalTraceEvents(data_provider_, timeline_));
  ASSERT_EQ(timeline_.timeline_data().entry_names.size(), 1u);
  EXPECT_EQ(timeline_.timeline_data().entry_names[0], "refetch_window_op");
}
#endif  // __EMSCRIPTEN__

TEST(TracePbEventParserIncrementalTest,
     StreamingStepChunksLargeSeriesAndLocksFullTimespanViewport) {
  IncrementalCacheCleanup cleanup_guard;
  // Advance synthetic clock by 10 ms on each read so an 8 ms budget yields
  // deterministically after each 32,768-event slice regardless of CPU speed.
  absl::Time fake_now = absl::UnixEpoch() + absl::Seconds(10);
  SetIncrementalStepTimeBudgetForTest(absl::Milliseconds(8));
  SetIncrementalStepClockForTest([&fake_now]() {
    const absl::Time current = fake_now;
    fake_now += absl::Milliseconds(10);
    return current;
  });

  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_incremental_fetch_enabled(true);
  DataProvider data_provider;

  // Frame 1 spans [0, 49 ms] inside a [0, 500 ms] full_timespan across 70,000
  // complete events (> kMaxTimedSeriesSliceEvents = 32,768), with a wide parent
  // spanning across index 32,768 to verify cross-slice parent nesting.
  xprof::TraceDataResponse frame1;
  frame1.set_full_timespan_start_ps(0ULL);
  frame1.set_full_timespan_end_ps(500'000'000'000ULL);  // 500 ms
  frame1.add_interned_strings("");
  frame1.add_interned_strings("stream_op");
  frame1.add_interned_strings("cross_slice_parent");
  auto* proc = frame1.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("device:TPU:0");
  auto* th = proc->add_threads();
  th->set_id(1);
  th->set_name("XLA Ops");
  auto* series = frame1.add_complete_events();
  series->mutable_metadata()->set_process_id(1);
  series->mutable_metadata()->set_thread_id(1);
  constexpr int kEvents = 70000;
  constexpr int kParentIdx = 32760;
  constexpr uint64_t kStepPs = 700'000ULL;
  // Event `i` starts at `(i + 1) * kStepPs` because delta[0] == kStepPs.
  constexpr uint64_t kParentEndPs = (32776ULL + 1) * kStepPs;
  for (int i = 0; i < kEvents; ++i) {
    series->add_deltas(kStepPs);
    if (i == kParentIdx) {
      series->add_durations((32776 - kParentIdx) * kStepPs);
      series->add_name_refs(2);
    } else {
      series->add_durations(500'000ULL);
      series->add_name_refs(1);
    }
    series->add_event_metadata();
  }

  // Frame 2 arrives later near 400 ms; viewport must stay locked at [0, 500 ms]
  // without rescaling horizontally.
  xprof::TraceDataResponse frame2;
  frame2.add_interned_strings("late_op");
  auto* s2 = frame2.add_complete_events();
  s2->mutable_metadata()->set_process_id(1);
  s2->mutable_metadata()->set_thread_id(1);
  s2->add_deltas(400'000'000'000ULL);  // 400 ms
  s2->add_durations(1'000'000'000ULL);
  // Frame 2's cumulative string index is 3 ([0]="", [1]="stream_op",
  // [2]="cross_slice_parent").
  s2->add_name_refs(3);
  s2->add_event_metadata();

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  absl::Status status = AppendCompressedXChunkFrame(frame1, xchn_payload);
  ASSERT_TRUE(status.ok()) << status;
  status = AppendCompressedXChunkFrame(frame2, xchn_payload);
  ASSERT_TRUE(status.ok()) << status;

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());

  // Pass a large event budget (> 32,768) so the 32,768 slice cap + 8 ms clock
  // deterministically split Frame 1 across multiple steps before flushing.
  int frame1_steps = 0;
  bool observed_mid_stream_flush = false;
  while (StepIncrementalTraceEvents(65536, data_provider, timeline)) {
    if (!observed_mid_stream_flush) {
      ++frame1_steps;
      if (HasFlushedIncrementalTraceEvents()) {
        observed_mid_stream_flush = true;
        EXPECT_EQ(timeline.visible_range(), TimeRange(0.0, 500000.0));
      }
    }
    ASSERT_LT(frame1_steps, 20);
  }
  EXPECT_GT(frame1_steps, 1);
  EXPECT_TRUE(observed_mid_stream_flush);
  EXPECT_EQ(timeline.visible_range(), TimeRange(0.0, 500000.0));

  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_EQ(timeline.visible_range(), TimeRange(0.0, 500000.0));
  EXPECT_TRUE(HasCachedTraceEventsIncremental());

  // Verify no overview child rect crossed the cross-slice parent end time.
  const double parent_end_us = kParentEndPs / 1000000.0;
  const FlameChartTimelineData& overview = timeline.timeline_data();
  for (size_t i = 0; i < overview.entry_names.size(); ++i) {
    if (overview.entry_names[i] == "cross_slice_parent") continue;
    if (overview.entry_start_times[i] < parent_end_us) {
      EXPECT_LE(overview.entry_start_times[i] + overview.entry_total_times[i],
                parent_end_us);
    }
  }

  // Refetch across the 32,768 boundary ([22.5, 23.5] ms) and frame1 tail
  // ([48.99, 49.01] ms) plus frame2 ([399.0, 405.0] ms).
  BeginIncrementalTraceEvents(std::make_pair(22.5, 23.5), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(10000, data_provider, timeline)) {
  }
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_THAT(timeline.timeline_data().entry_names, Contains("stream_op"));
  EXPECT_THAT(timeline.timeline_data().entry_names,
              Contains("cross_slice_parent"));

  BeginIncrementalTraceEvents(std::make_pair(48.99, 49.01), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(10000, data_provider, timeline)) {
  }
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_THAT(timeline.timeline_data().entry_names, Contains("stream_op"));

  BeginIncrementalTraceEvents(std::make_pair(399.0, 405.0), data_provider,
                              timeline);
  while (StepIncrementalTraceEvents(10000, data_provider, timeline)) {
  }
  FinishIncrementalTraceEvents(data_provider, timeline);
  EXPECT_THAT(timeline.timeline_data().entry_names, Contains("late_op"));
}

TEST(TracePbEventParserIncrementalTest,
     DeferredZoomFlushDoesNotAdvanceFlushedFrameCounter) {
  IncrementalCacheCleanup cleanup_guard;
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  DataProvider data_provider;

  auto make_frame = [](absl::string_view name, uint64_t ts_ps, bool with_meta) {
    xprof::TraceDataResponse frame;
    if (with_meta) {
      frame.add_interned_strings("");
      frame.set_full_timespan_start_ps(0ULL);
      frame.set_full_timespan_end_ps(100'000'000'000ULL);
      auto* proc = frame.mutable_metadata()->add_processes();
      proc->set_id(1);
      proc->set_name("device:TPU:0");
      auto* th = proc->add_threads();
      th->set_id(1);
      th->set_name("XLA Ops");
    }
    frame.add_interned_strings(std::string(name));
    auto* s = frame.add_complete_events();
    s->mutable_metadata()->set_process_id(1);
    s->mutable_metadata()->set_thread_id(1);
    s->add_deltas(ts_ps);
    s->add_durations(1'000'000ULL);
    s->add_name_refs(with_meta ? 1 : 2);
    s->add_event_metadata();
    return frame;
  };

  std::string xchn_payload(tensorflow::profiler::kXChunkMagic);
  ASSERT_TRUE(AppendCompressedXChunkFrame(
                  make_frame("op_f1", 1'000'000ULL, true), xchn_payload)
                  .ok());
  ASSERT_TRUE(AppendCompressedXChunkFrame(
                  make_frame("op_f2", 2'000'000ULL, false), xchn_payload)
                  .ok());
  ASSERT_TRUE(AppendCompressedXChunkFrame(
                  make_frame("op_f3", 3'000'000ULL, false), xchn_payload)
                  .ok());
  ASSERT_TRUE(AppendCompressedXChunkFrame(
                  make_frame("op_f4", 4'000'000ULL, false), xchn_payload)
                  .ok());

  BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
  AppendCompressedTraceEventsIncremental(
      reinterpret_cast<uintptr_t>(xchn_payload.data()), xchn_payload.size());

  // Frame 1 flushes normally (frames_flushed == 1).
  EXPECT_TRUE(StepIncrementalTraceEvents(100, data_provider, timeline));
  EXPECT_TRUE(HasFlushedIncrementalTraceEvents());
  ASSERT_EQ(timeline.timeline_data().entry_names.size(), 1u);

  // Start an animated zoom so Frames 2 and 3 defer their flushes.
  timeline.SetVisibleRange({1000.0, 5000.0}, /*animate=*/true);
  ASSERT_NE(timeline.visible_range(), timeline.visible_range_target());
  EXPECT_TRUE(StepIncrementalTraceEvents(100, data_provider, timeline));
  EXPECT_TRUE(StepIncrementalTraceEvents(100, data_provider, timeline));
  EXPECT_EQ(timeline.timeline_data().entry_names.size(), 1u);

  // Settle the zoom animation before Frame 4: because deferred flushes did not
  // increment frames_flushed, Frame 4 still auto-flushes (frames_flushed < 3)
  // even if more frames were queued.
  timeline.SetVisibleRange({1000.0, 5000.0}, /*animate=*/false);
  EXPECT_FALSE(StepIncrementalTraceEvents(100, data_provider, timeline));
  EXPECT_EQ(timeline.timeline_data().entry_names.size(), 4u);
  FinishIncrementalTraceEvents(data_provider, timeline);
}

TEST(TracePbEventParserIncrementalTest,
     StepDeadlineAppliesToAsyncAndCounterSeriesAndBypassesWhenUntimed) {
  IncrementalCacheCleanup cleanup_guard;
  absl::Time fake_now = absl::UnixEpoch() + absl::Seconds(10);
  SetIncrementalStepTimeBudgetForTest(absl::Milliseconds(8));
  SetIncrementalStepClockForTest([&fake_now]() {
    const absl::Time current = fake_now;
    fake_now += absl::Milliseconds(10);
    return current;
  });

  xprof::TraceDataResponse frame;
  frame.add_interned_strings("");
  frame.add_interned_strings("async_1");
  frame.add_interned_strings("async_2");
  frame.add_interned_strings("ctr_1");
  frame.add_interned_strings("ctr_2");
  auto* proc = frame.mutable_metadata()->add_processes();
  proc->set_id(1);
  proc->set_name("device:TPU:0");
  for (int s = 0; s < 2; ++s) {
    auto* async_s = frame.add_async_events();
    async_s->mutable_metadata()->set_process_id(1);
    async_s->mutable_metadata()->set_name_ref(1 + s);
    async_s->add_deltas(1'000'000ULL);
    async_s->add_durations(500'000ULL);
    async_s->add_event_metadata();

    auto* ctr_s = frame.add_counter_events();
    ctr_s->mutable_metadata()->set_process_id(1);
    ctr_s->mutable_metadata()->set_name_ref(3 + s);
    ctr_s->add_deltas(1'000'000ULL);
    ctr_s->add_event_metadata()->set_counter_value_double(42.0);
  }
  const absl::StatusOr<std::string> compressed =
      CompressTraceDataResponse(frame);
  ASSERT_TRUE(compressed.ok());

  // Timed step yields after the first async series once step_deadline passes.
  {
    ColorPalette palette = ColorPalette::Default();
    Timeline timeline(palette);
    DataProvider data_provider;
    BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
    AppendCompressedTraceEventsIncremental(
        reinterpret_cast<uintptr_t>(compressed->data()), compressed->size());
    EXPECT_TRUE(StepIncrementalTraceEvents(1000, data_provider, timeline,
                                           /*enforce_time_budget=*/true));
    while (StepIncrementalTraceEvents(1000, data_provider, timeline,
                                      /*enforce_time_budget=*/true)) {
    }
    FinishIncrementalTraceEvents(data_provider, timeline);
    EXPECT_EQ(timeline.timeline_data().entry_names.size(), 2u);
  }

  // Untimed step (background tab) drains the entire frame in one step.
  {
    ColorPalette palette = ColorPalette::Default();
    Timeline timeline(palette);
    DataProvider data_provider;
    BeginIncrementalTraceEvents(std::nullopt, data_provider, timeline);
    AppendCompressedTraceEventsIncremental(
        reinterpret_cast<uintptr_t>(compressed->data()), compressed->size());
    EXPECT_FALSE(StepIncrementalTraceEvents(1000, data_provider, timeline,
                                            /*enforce_time_budget=*/false));
    FinishIncrementalTraceEvents(data_provider, timeline);
    EXPECT_EQ(timeline.timeline_data().entry_names.size(), 2u);
  }
}

}  // namespace traceviewer
