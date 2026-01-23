#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/em_asm.h>
#else
// Stub EM_ASM for non-Emscripten builds to allow compilation in non-WASM
// environments (e.g. for host tests or IDE analysis).
#define EM_ASM(...)
#endif

#include "util/math/mathutil.h"

namespace traceviewer {

CanvasState CanvasState::instance_;
uint8_t CanvasState::current_version_ = 1;

CanvasState::CanvasState() {
  EM_ASM(
      {
        if (typeof window == 'undefined') return;
        const canvas = document.getElementById('canvas');
        if (!canvas) return;
        setValue($0, window.devicePixelRatio, 'float');
        setValue($1, canvas.clientWidth, 'i32');
        setValue($2, canvas.clientHeight, 'i32');
      },
      &device_pixel_ratio_, &width_, &height_);
}

const CanvasState& CanvasState::Current() { return instance_; }

void CanvasState::SetState(float dpr, int32_t width, int32_t height) {
  const CanvasState new_state(dpr, width, height);

  if (instance_ == new_state) return;

  instance_ = new_state;
  current_version_++;
}

bool CanvasState::operator==(const CanvasState& other) const {
  return width_ == other.width_ && height_ == other.height_ &&
         MathUtil::AlmostEquals(device_pixel_ratio_, other.device_pixel_ratio_);
}

}  // namespace traceviewer
