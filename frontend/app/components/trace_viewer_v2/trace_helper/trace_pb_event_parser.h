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
//
// Ownership: `data_ptr` is a WASM heap pointer allocated by JS (`_malloc`).
// The callee only reads the buffer; the JS caller must free it after return
// (see `withWasmHeapBuffer` in wasm_string_utils.ts).
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

// Parses compressed protobuf search results from a WASM heap buffer and
// updates the timeline search state.
//
// Ownership: same as ParseAndProcessCompressedTraceEvents — JS allocates with
// `_malloc` and must `_free` after the call returns.
void SetCompressedSearchResultsInWasm(uintptr_t data_ptr, size_t data_size);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_
