#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_CANVAS_STATE_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_CANVAS_STATE_H_

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>  // NO_LINT
#else
#define EM_ASM(...)
#endif

#include <cstdint>
#include <optional>
#include <type_traits>

#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

// The CanvasState class is responsible for tracking the state of the HTML
// canvas element, particularly its dimensions (width and height) and the device
// pixel ratio. This information is crucial for rendering graphics correctly,
// especially when the browser window is resized or moved between screens with
// different DPI settings. The class provides methods to get the current state
// and detect changes.
class CanvasState {
 public:
  static const CanvasState& Current();
  // Checks for changes in canvas state and updates the current instance if
  // needed. Returns true if the state was updated, false otherwise.
  static bool Update();
  static uint8_t version() { return current_version_; }

  bool operator==(const CanvasState& other) const;

  // Returns the canvas dimensions in physical pixels. Physical pixels are the
  // actual hardware pixels on the screen. High-DPI screens (like Retina
  // displays) pack more physical pixels into the same area, resulting in
  // sharper images.
  ImVec2 physical_pixels() const {
    return {width_ * device_pixel_ratio_, height_ * device_pixel_ratio_};
  }

  // Returns the canvas dimensions in logical pixels. Logical pixels are
  // resolution-independent units used in web development (aka CSS pixels).
  // They are scaled by the device pixel ratio to map to physical pixels. For
  // example, on a screen with a DPR of 2.0, one logical pixel corresponds to a
  // 2x2 block of physical pixels.
  ImVec2 logical_pixels() const {
    return {static_cast<float>(width_), static_cast<float>(height_)};
  }

  // Returns the device pixel ratio (DPR), which is the ratio of physical
  // pixels to logical pixels. For example, a DPR of 2.0 means there are 2
  // physical pixels for every 1 logical pixel in each dimension (2x width, 2x
  // height), totaling 4 physical pixels for every 1 logical pixel.
  float device_pixel_ratio() const { return device_pixel_ratio_; }

  // Returns canvas height in logical pixels (aka CSS pixels), derived from
  // canvas.clientHeight.
  int32_t height() const { return height_; }
  // Returns canvas width in logical pixels (aka CSS pixels), derived from
  // canvas.clientWidth.
  int32_t width() const { return width_; }

 private:
  friend class CanvasStateTest;

  CanvasState();
  CanvasState(float dpr, int32_t height, int32_t width)
      : device_pixel_ratio_(dpr), height_(height), width_(width) {}

  static CanvasState instance_;
  static uint8_t current_version_;

  float device_pixel_ratio_ = 1.0f;
  int32_t height_ = 0;
  int32_t width_ = 0;
};

// The DprAware class is used to wrap numerical values that need to be adjusted
// based on the screen's device pixel ratio (DPR). This is useful for
// ensuring that UI elements are rendered at the correct size across
// different displays.
template <typename T>
class DprAware {
 public:
  static_assert(std::is_arithmetic_v<T>, "DprAware<T> must be numeric");

  // A default constructed DprAware has a value of 0.
  DprAware() : DprAware(0) {}
  explicit DprAware(T value)
      : value_(value), cache_(0), cache_version_(std::nullopt) {}

  explicit operator float() const {
    const uint8_t current = CanvasState::version();
    if (cache_version_.has_value() && *cache_version_ == current) {
      return cache_;
    }
    cache_ = value_ * CanvasState::Current().device_pixel_ratio();
    cache_version_ = current;
    return cache_;
  }

  float operator*() const { return static_cast<float>(*this); }

  constexpr T value() const { return value_; }

 private:
  T value_;
  mutable float cache_;
  mutable std::optional<uint8_t> cache_version_;
};

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_CANVAS_STATE_H_
