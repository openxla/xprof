#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

class DataProvider {
 public:
  // Returns a list of flow categories present in the trace.
  const std::vector<int>& GetFlowCategories() const;

  // Returns process mappings (pid -> hostname).
  absl::flat_hash_map<ProcessId, std::string> GetProcessMappings() const;

  // Processes vectors of TraceEvent structs.
  void ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                          Timeline& timeline);

  // Processes protobuf TraceDataResponse directly using arena-backed
  // ProtoEventRef.
  void ProcessTraceEvents(const xprof::TraceDataResponse& response,
                          Timeline& timeline,
                          std::optional<std::pair<Milliseconds, Milliseconds>>
                              visible_range_from_url = std::nullopt);

 private:
  std::vector<int> present_flow_categories_;
  absl::flat_hash_map<ProcessId, std::string> process_names_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
