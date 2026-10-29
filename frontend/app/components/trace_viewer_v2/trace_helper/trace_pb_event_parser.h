#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_

#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>

namespace traceviewer {

class DataProvider;
class Timeline;

// Parses compressed trace data and updates the timeline in the Application
// instance. This is the main entry point for processing compressed trace data
// from the frontend.
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url);

// Overload of ParseAndProcessCompressedTraceEvents that accepts explicitly
// injected dependencies (DataProvider and Timeline) to enable testability
// without relying on a global Application instance.
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline);

// Begins an incremental trace loading session.
void BeginIncrementalTraceEvents(const emscripten::val& visible_range_from_url,
                                 DataProvider& data_provider,
                                 Timeline& timeline);
void BeginIncrementalTraceEvents(const emscripten::val& visible_range_from_url);

// Queues a compressed TraceDataResponse frame (or an "XCHN"-prefixed sequence
// of length-delimited compressed frames) for incremental processing.
void AppendCompressedTraceEventsIncremental(uintptr_t data_ptr,
                                            size_t data_size);

// Decompresses and unmarshals up to `max_events_per_step` events (at process
// boundaries) and appends them incrementally to the timeline. Returns true if
// more events or queued frames remain to be processed.
bool StepIncrementalTraceEvents(size_t max_events_per_step,
                                DataProvider& data_provider,
                                Timeline& timeline);
bool StepIncrementalTraceEvents(size_t max_events_per_step);

// Flushes any remaining queued incremental trace events and finalizes the
// incremental loading session. Returns true if the full trace was loaded
// without compaction.
bool FinishIncrementalTraceEvents(DataProvider& data_provider,
                                  Timeline& timeline);
bool FinishIncrementalTraceEvents();

// Cancels an in-flight incremental loading session without applying any
// pending buffered refetch events to the timeline.
void CancelIncrementalTraceEvents(DataProvider& data_provider,
                                  Timeline& timeline);
void CancelIncrementalTraceEvents();

// Restores the stashed full-trace overview in O(1) time when zooming back out
// from a filtered window, avoiding re-decompression and re-parsing.
bool RestoreFullTraceOverviewIfAvailable(Timeline& timeline);
bool RestoreFullTraceOverviewIfAvailable();

// Returns true if the next compressed frame in sequence overlaps the current
// session's filter window (or if no frame summary is available yet). When this
// returns false, the caller may skip copying and appending the frame, as the
// parser has already advanced its frame index past it.
bool ShouldEnqueueCompressedFrameIncremental();

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_
