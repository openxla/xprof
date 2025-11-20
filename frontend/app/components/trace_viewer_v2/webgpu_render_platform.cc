#include "xprof/frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"

#include <webgpu/webgpu.h>  // NO_LINT
#include <webgpu/webgpu_cpp.h>

#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"
#include "xprof/frontend/app/components/trace_viewer_v2/imgui_webgpu_backend.h"
#include "absl/log/check.h"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

void WGPURenderPlatform::Init(const CanvasState& canvas_state) {
  instance_ = wgpu::CreateInstance();

  device_ = wgpu::Device::Acquire(emscripten_webgpu_get_device());
  CHECK(device_) << "Error creating device.";
  queue_ = device_.GetQueue();

  wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc;
  canvas_desc.selector = "#canvas";
  wgpu::SurfaceDescriptor surface_desc;
  surface_desc.nextInChain = &canvas_desc;

  surface_ = instance_.CreateSurface(&surface_desc);
  CHECK(surface_) << "Error when creating canvas surface.";

  wgpu::SurfaceCapabilities capabilities;
  surface_.GetCapabilities(adapter_, &capabilities);
  surface_config_.device = device_;
  surface_config_.format = capabilities.formats[0];
  surface_config_.presentMode = wgpu::PresentMode::Fifo;
  surface_config_.usage = wgpu::TextureUsage::RenderAttachment;
  surface_config_.alphaMode = capabilities.alphaModes[0];

  ResizeSurface(canvas_state);

  ImGui_ImplWGPU_InitInfo imgui_info = {};
  imgui_info.device = device_;
  imgui_info.num_frames_in_flight = 3;
  imgui_info.target_format = capabilities.formats[0];
  imgui_info.depth_stencil_format = wgpu::TextureFormat::Undefined;
  imgui_info.multisample_state.count = kMultisampleCount;

  CHECK(ImGui_ImplWGPU_Init(&imgui_info))
      << "Failed to initialize the ImGUI WebGPU backend.";
}

void WGPURenderPlatform::ResizeSurface(const CanvasState& canvas_state) {
  ImGui_ImplWGPU_InvalidateDeviceObjects();

  const ImVec2 size = canvas_state.physical_pixels();
  // WebGPU surface configuration requires dimensions in physical pixels.
  // This ensures the underlying framebuffer is sized correctly for the
  // screen's resolution, preventing errors caused by mismatches between
  // ImGui's scaled rendering output and the framebuffer size.
  surface_config_.width = size.x;
  surface_config_.height = size.y;
  surface_.Configure(&surface_config_);

  InitMultisampleTexture();
}

void WGPURenderPlatform::InitMultisampleTexture() {
  wgpu::TextureDescriptor multisample_tex_desc{};
  multisample_tex_desc.label = "ImGui multisample texture";
  multisample_tex_desc.size = {surface_config_.width, surface_config_.height,
                               1};
  multisample_tex_desc.format = surface_config_.format;
  multisample_tex_desc.sampleCount = kMultisampleCount;
  multisample_tex_desc.usage = wgpu::TextureUsage::RenderAttachment;
  multisample_tex_ = device_.CreateTexture(&multisample_tex_desc);
  CHECK(multisample_tex_) << "Error creating multisample texture.";
}

void WGPURenderPlatform::NewFrame() {
  ImGui_ImplWGPU_NewFrame();
  ImGui::NewFrame();
}

void WGPURenderPlatform::RenderFrame() {
  wgpu::SurfaceTexture surface_tex;
  surface_.GetCurrentTexture(&surface_tex);

  wgpu::RenderPassColorAttachment color_attach{};
  color_attach.view = multisample_tex_.CreateView();
  color_attach.resolveTarget = surface_tex.texture.CreateView();
  color_attach.loadOp = wgpu::LoadOp::Clear;
  color_attach.storeOp = wgpu::StoreOp::Store;
  color_attach.clearValue = {20 / 255.f, 20 / 255.f, 17 / 255.f, 1.f};

  wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();

  wgpu::RenderPassDescriptor render_pass_desc{};
  render_pass_desc.label = "ImGui render pass descriptor";
  render_pass_desc.colorAttachmentCount = 1;
  render_pass_desc.colorAttachments = &color_attach;
  render_pass_desc.depthStencilAttachment = nullptr;
  wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&render_pass_desc);

  ImGui::Render();
  ImDrawData* draw_data = ImGui::GetDrawData();
  ImGui_ImplWGPU_RenderDrawData(draw_data, pass);
  pass.End();

  wgpu::CommandBufferDescriptor cmd_buffer_desc{};
  wgpu::CommandBuffer cmd_buffer = encoder.Finish(&cmd_buffer_desc);
  queue_.Submit(1, &cmd_buffer);
}

}  // namespace traceviewer
