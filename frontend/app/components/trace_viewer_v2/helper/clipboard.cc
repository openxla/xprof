#include "frontend/app/components/trace_viewer_v2/helper/clipboard.h"

#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include "absl/strings/string_view.h"

namespace traceviewer {

void CopyToClipboard(absl::string_view text) {
#ifdef __EMSCRIPTEN__
  std::string s(text);
  EM_ASM(
      {
        const textToCopy = UTF8ToString($0);
        if (navigator.clipboard) {
          navigator.clipboard.writeText(textToCopy).catch(function(err){});
        } else {
          const textArea = document.createElement("textarea");
          textArea.value = textToCopy;
          textArea.style.position = "fixed";
          document.body.appendChild(textArea);
          textArea.focus();
          textArea.select();
          try {
            document.execCommand("copy");
          } catch (err) {
          }
          document.body.removeChild(textArea);
        }
      },
      s.c_str());
#endif
}

}  // namespace traceviewer
