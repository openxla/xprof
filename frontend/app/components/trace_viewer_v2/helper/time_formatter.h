#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_HELPER_TIME_FORMATTER_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_HELPER_TIME_FORMATTER_H_

#include <string>

#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

std::string FormatTime(Microseconds time_us);

// Calculates a "nice" time interval for ruler ticks. The interval is chosen
// from a set of values {1, 2, 5} scaled by a power of 10, ensuring it is
// greater than or equal to `min_interval`.
Microseconds CalculateNiceInterval(Microseconds min_interval);

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_HELPER_TIME_FORMATTER_H_
