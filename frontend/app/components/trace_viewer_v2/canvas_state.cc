#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"

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
        setValue($1, canvas.clientHeight, 'i32');
        setValue($2, canvas.clientWidth, 'i32');
      },
      &device_pixel_ratio_, &height_, &width_);
}

const CanvasState& CanvasState::Current() { return instance_; }

bool CanvasState::Update() {
  const CanvasState new_state;
  if (instance_ == new_state) return false;
  instance_ = new_state;
  current_version_++;
  return true;
}

bool CanvasState::operator==(const CanvasState& other) const {
  return height_ == other.height_ && width_ == other.width_ &&
         MathUtil::AlmostEquals(device_pixel_ratio_, other.device_pixel_ratio_);
}

}  // namespace traceviewer
