#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_WEBGPU_RENDER_PLATFORM_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_WEBGPU_RENDER_PLATFORM_H_

#include <cstdint>
#include <memory>

#include "webgpu/webgpu_cpp.h"

#include "frontend/app/components/trace_viewer_v2/canvas_state.h"

namespace traceviewer {

class WGPURenderPlatform {
 public:
  ~WGPURenderPlatform();
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

  bool is_offscreen_ = false;
  wgpu::TextureFormat target_format_ = wgpu::TextureFormat::RGBA8Unorm;
  wgpu::Texture offscreen_target_tex_ = nullptr;
  std::shared_ptr<uint64_t> latest_frame_id_ = std::make_shared<uint64_t>(0);
  uint32_t width_ = 0;
  uint32_t height_ = 0;

  void InitMultisampleTexture();
};
}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_WEBGPU_RENDER_PLATFORM_H_
