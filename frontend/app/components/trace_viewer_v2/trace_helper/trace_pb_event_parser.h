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
// from the frontend. When `free_data_ptr` is true, takes ownership of the
// WASM heap buffer at `data_ptr` and frees it immediately after decompression.
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, bool free_data_ptr = false);

// Overload of ParseAndProcessCompressedTraceEvents that accepts explicitly
// injected dependencies (DataProvider and Timeline) to enable testability
// without relying on a global Application instance.
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline, bool free_data_ptr = false);

// Starts or resets a streaming trace ingestion session (`P1`–`P12`).
void BeginTraceStream(uint32_t stream_id = 0);

// Pushes a chunk of framed compressed stream bytes from WASM linear memory at
// `[data_ptr, data_ptr + data_size)`. Returns true if all complete frames in
// the buffer were valid and decompressed/ingested without error. Does NOT take
// ownership of `data_ptr` (allowing JS to reuse a single staging buffer across
// network chunks).
bool PushTraceStreamBytes(uintptr_t data_ptr, size_t data_size);

// Verifies the stream trailer (`P10`) and commits the streamed trace into
// `data_provider` and `timeline`. Returns false and leaves `data_provider` and
// `timeline` untouched if the stream is truncated or fails trailer checksum
// verification, enabling fallback to the monolithic response.
bool FinishTraceStream(const emscripten::val& visible_range_from_url,
                       DataProvider& data_provider, Timeline& timeline);

bool FinishTraceStream(const emscripten::val& visible_range_from_url);

// Aborts any active streaming trace ingestion session and frees its buffers.
void AbortTraceStream();

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_
