#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_WEBGPU_RENDER_PLATFORM_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_WEBGPU_RENDER_PLATFORM_H_

#include <webgpu/webgpu_cpp.h>

#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"

namespace traceviewer {

class WGPURenderPlatform {
 public:
  void Init(const CanvasState& canvas_state);
  void ResizeSurface(const CanvasState& canvas_state);
  void NewFrame();
  void RenderFrame();

 private:
  static constexpr int kMultisampleCount = 4;
  wgpu::Instance instance_ = nullptr;
  wgpu::Adapter adapter_ = nullptr;
  wgpu::Device device_ = nullptr;
  wgpu::Queue queue_ = nullptr;
  wgpu::Surface surface_ = nullptr;
  wgpu::SurfaceConfiguration surface_config_ = {};
  wgpu::Texture multisample_tex_ = nullptr;

  void InitMultisampleTexture();
};
}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_WEBGPU_RENDER_PLATFORM_H_
