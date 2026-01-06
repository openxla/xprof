#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_EVENT_DATA_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_EVENT_DATA_H_

#include <any>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace traceviewer {

// A generic dictionary type for event data, using absl::any to hold
// heterogeneous value types. This is used to construct event payloads that are
// later converted to JavaScript objects.
using EventData = absl::flat_hash_map<std::string, std::any>;

// Constants used for defining event names and data keys for interop with
// JavaScript.

// Following constants are used for event selected event.
inline constexpr absl::string_view kEventSelected = "eventselected";

inline constexpr absl::string_view kEventSelectedIndex = "eventIndex";
inline constexpr absl::string_view kEventSelectedName = "name";

// Constants for fetch data event.
inline constexpr absl::string_view kFetchData = "fetch_data";

inline constexpr absl::string_view kFetchDataStart = "start_time_ms";
inline constexpr absl::string_view kFetchDataEnd = "end_time_ms";

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_EVENT_DATA_H_
