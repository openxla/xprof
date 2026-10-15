#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/fingerprint.h"

namespace traceviewer {

EventId GenerateEventId(absl::string_view name, Microseconds ts,
                        Microseconds dur) {
  // We use Fingerprint64 for a stable event ID because absl::HashOf
  // does not guarantee stability across different executions or binaries,
  // and we need consistency for event associations.
  const int64_t ts_ps = static_cast<int64_t>(std::round(ts * 1000000.0));
  const int64_t dur_ps = static_cast<int64_t>(std::round(dur * 1000000.0));
  char buf[512];
  if (name.size() + 44 <= sizeof(buf)) {
    std::memcpy(buf, name.data(), name.size());
    char* p = buf + name.size();
    *p++ = ':';
    p = std::to_chars(p, buf + sizeof(buf), ts_ps).ptr;
    *p++ = ':';
    p = std::to_chars(p, buf + sizeof(buf), dur_ps).ptr;
    return tsl::Fingerprint64(
        absl::string_view(buf, static_cast<size_t>(p - buf)));
  }
  return tsl::Fingerprint64(absl::StrCat(name, ":", ts_ps, ":", dur_ps));
}

uint64_t ParseFlowIdString(absl::string_view id_str) {
  if (id_str.empty()) return 0;
  uint64_t flow_id = 0;
  if (absl::SimpleAtoi(id_str, &flow_id)) {
    return flow_id;
  }
  if (absl::StartsWithIgnoreCase(id_str, "0x") &&
      absl::SimpleHexAtoi(id_str, &flow_id)) {
    return flow_id;
  }
  const uint64_t fp = tsl::Fingerprint64(id_str);
  return fp != 0 ? fp : 1;
}

}  // namespace traceviewer
