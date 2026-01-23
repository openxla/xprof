#ifndef THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_TRACE_PROCESSOR_H_
#define THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_TRACE_PROCESSOR_H_

#include <cstdint>

#include "absl/base/nullability.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {

// This class processes an XprofTrace in-place to prepare it for conversion to
// Perfetto trace format. It focuses on better visualization of Megascale
// actions and their dependencies.
class TraceProcessor {
 public:
  explicit TraceProcessor(XprofTrace* absl_nonnull trace) : trace_(*trace) {}

  // This function makes a series of modifications to the trace in-place:
  // - Reorders events in a more logical order.
  // - Groups Megascale events with their corresponding TPU events.
  // - Adds flows between key actions in the Megascale action graph.
  void Process();

 private:
  // Sorts events in each track by timestamp.
  void SortEvents();
  // Assigns run IDs to events in each track.
  // This is used to group events that are part of the same XLA program run. We
  // don't assign run IDs to program runs that were not fully captured in the
  // profile.
  void AssignRunIds();
  // Marks the last H2D events for each execution event. This is needed for
  // adding H2D -> recv-done flows.
  void MarkLastH2DEvents();
  // Resolves flows between TPU events and Megascale events.
  void ResolveFlows();
  // Adds a counter track for network metrics.
  void AddNetworkCounters();
  // Modifies track names to make them more readable and to control ordering in
  // Perfetto UI.
  void ModifyTrackNames();

  XprofTrace& trace_;

  // Helper for generating unique flow IDs during processing
  int64_t next_flow_id_ = 1;
};

}  // namespace xprof::megascale

#endif  // THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_TRACE_PROCESSOR_H_
