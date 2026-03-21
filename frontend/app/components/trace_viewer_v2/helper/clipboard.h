#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_HELPER_CLIPBOARD_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_HELPER_CLIPBOARD_H_

#include "absl/strings/string_view.h"

namespace traceviewer {

// Copies text to the system clipboard.
// Supports WebAssembly environments by invoking browser navigator.clipboard.
void CopyToClipboard(absl::string_view text);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_HELPER_CLIPBOARD_H_
