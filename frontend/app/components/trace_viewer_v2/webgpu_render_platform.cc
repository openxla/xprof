#include "frontend/app/components/trace_viewer_v2/webgpu_render_platform.h"

#include <emscripten.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#ifdef __EMSCRIPTEN__
#include <emscripten/em_asm.h>

#include "webgpu/webgpu.h"
extern "C" WGPUDevice emscripten_webgpu_get_device();
#endif
#include "webgpu/webgpu_cpp.h"

#include "frontend/app/components/trace_viewer_v2/canvas_state.h"
#include "frontend/app/components/trace_viewer_v2/imgui_webgpu_backend.h"
#include "absl/log/check.h"
#include "imgui.h"

namespace traceviewer {

WGPURenderPlatform::~WGPURenderPlatform() {
  ++(*latest_frame_id_);
  ImGui_ImplWGPU_Shutdown();
}

void WGPURenderPlatform::Init(const CanvasState& canvas_state) {
  instance_ = wgpu::CreateInstance();

  device_ = wgpu::Device::Acquire(emscripten_webgpu_get_device());
  CHECK(device_) << "Error creating device.";
  queue_ = device_.GetQueue();

  int is_offscreen = 0;
  // clang-format off
  // The JS below must not be reflowed: a line break inside `===` splits it
  // into `== =`, which emcc rejects.
  EM_ASM(
      {
        const canvas = document.getElementById('canvas');
        const offscreen = !canvas || canvas.dataset.offscreen == 'true';
        setValue($0, offscreen ? 1 : 0, 'i32');
      },
      &is_offscreen);
  // clang-format on
  is_offscreen_ = (is_offscreen != 0);

  wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc;
  canvas_desc.selector = "#canvas";
  if (!is_offscreen_) {
    wgpu::SurfaceDescriptor surface_desc;
    surface_desc.nextInChain = &canvas_desc;
    surface_ = instance_.CreateSurface(&surface_desc);
    CHECK(surface_) << "Error when creating canvas surface.";

    wgpu::SurfaceCapabilities capabilities;
    surface_.GetCapabilities(adapter_, &capabilities);
    target_format_ = capabilities.formats[0];
    surface_config_.device = device_;
    surface_config_.format = target_format_;
    surface_config_.presentMode = wgpu::PresentMode::Fifo;
    surface_config_.usage = wgpu::TextureUsage::RenderAttachment;
    surface_config_.alphaMode = capabilities.alphaModes[0];
  } else {
    target_format_ = wgpu::TextureFormat::RGBA8Unorm;
  }

  ResizeSurface(canvas_state);

  ImGui_ImplWGPU_InitInfo imgui_info = {};
  imgui_info.device = device_;
  imgui_info.num_frames_in_flight = 3;
  imgui_info.target_format = target_format_;
  imgui_info.depth_stencil_format = wgpu::TextureFormat::Undefined;
  imgui_info.multisample_state.count = kMultisampleCount;

  CHECK(ImGui_ImplWGPU_Init(&imgui_info))
      << "Failed to initialize the ImGUI WebGPU backend.";
}

void WGPURenderPlatform::ResizeSurface(const CanvasState& canvas_state) {
  const ImVec2 size = canvas_state.physical_pixels();
  // WebGPU surface configuration requires dimensions in physical pixels.
  // This ensures the underlying framebuffer is sized correctly for the
  // screen's resolution, preventing errors caused by mismatches between
  // ImGui's scaled rendering output and the framebuffer size.
  if (size.x <= 0.0f || size.y <= 0.0f) return;
  const uint32_t new_width = static_cast<uint32_t>(size.x);
  const uint32_t new_height = static_cast<uint32_t>(size.y);
  if (new_width == 0 || new_height == 0) return;
  width_ = new_width;
  height_ = new_height;
  ++(*latest_frame_id_);

  if (!is_offscreen_) {
    surface_config_.width = width_;
    surface_config_.height = height_;
    surface_.Configure(&surface_config_);
  } else {
    wgpu::TextureDescriptor target_desc{};
    target_desc.label = "ImGui offscreen resolve target";
    target_desc.size = {width_, height_, 1};
    target_desc.format = target_format_;
    target_desc.usage =
        wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
    offscreen_target_tex_ = device_.CreateTexture(&target_desc);
    CHECK(offscreen_target_tex_) << "Error creating offscreen target texture.";
  }

  // Note: We do NOT call ImGui_ImplWGPU_InvalidateDeviceObjects() here.
  // Doing so would destroy and recreate all WebGPU resources (like the font
  // texture) on every resize, causing significant flickering.

  InitMultisampleTexture();
}

void WGPURenderPlatform::InitMultisampleTexture() {
  wgpu::TextureDescriptor multisample_tex_desc{};
  multisample_tex_desc.label = "ImGui multisample texture";
  multisample_tex_desc.size = {width_, height_, 1};
  multisample_tex_desc.format = target_format_;
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
  if (width_ == 0 || height_ == 0 || !multisample_tex_ ||
      (is_offscreen_ && !offscreen_target_tex_)) {
    ImGui::EndFrame();
    return;
  }

  wgpu::SurfaceTexture surface_tex;
  wgpu::TextureView resolve_target_view = nullptr;
  if (!is_offscreen_) {
    surface_.GetCurrentTexture(&surface_tex);
    resolve_target_view = surface_tex.texture.CreateView();
  } else {
    resolve_target_view = offscreen_target_tex_.CreateView();
  }

  wgpu::RenderPassColorAttachment color_attach{};
  color_attach.view = multisample_tex_.CreateView();
  color_attach.resolveTarget = resolve_target_view;
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

  wgpu::Buffer readback_buffer = nullptr;
  uint32_t bytes_per_row = 0;
  if (is_offscreen_) {
    bytes_per_row = (width_ * 4 + 255) & ~255;
    wgpu::BufferDescriptor buf_desc{};
    buf_desc.label = "ImGui offscreen readback buffer";
    buf_desc.size = static_cast<uint64_t>(bytes_per_row) * height_;
    buf_desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readback_buffer = device_.CreateBuffer(&buf_desc);
    CHECK(readback_buffer) << "Error creating readback buffer.";

    wgpu::TexelCopyTextureInfo copy_src{.texture = offscreen_target_tex_};
    wgpu::TexelCopyBufferInfo copy_dst{
        .layout = {.bytesPerRow = bytes_per_row, .rowsPerImage = height_},
        .buffer = readback_buffer};
    wgpu::Extent3D copy_size = {
        .width = width_, .height = height_, .depthOrArrayLayers = 1};
    encoder.CopyTextureToBuffer(&copy_src, &copy_dst, &copy_size);
  }

  wgpu::CommandBufferDescriptor cmd_buffer_desc{};
  wgpu::CommandBuffer cmd_buffer = encoder.Finish(&cmd_buffer_desc);
  queue_.Submit(1, &cmd_buffer);

  if (is_offscreen_) {
    const uint64_t frame_id = ++(*latest_frame_id_);
    readback_buffer.MapAsync(
        wgpu::MapMode::Read, 0, wgpu::kWholeMapSize,
        wgpu::CallbackMode::AllowSpontaneous,
        [readback_buffer, latest_frame_id = latest_frame_id_, frame_id,
         width = width_, height = height_,
         bytes_per_row](wgpu::MapAsyncStatus status, wgpu::StringView message) {
          if (status != wgpu::MapAsyncStatus::Success ||
              frame_id != *latest_frame_id) {
            readback_buffer.Destroy();
            return;
          }
          const void* mapped = readback_buffer.GetConstMappedRange(
              0, static_cast<size_t>(bytes_per_row) * height);
          if (mapped == nullptr) {
            readback_buffer.Unmap();
            readback_buffer.Destroy();
            return;
          }
          // clang-format off
          EM_ASM(
              {
                const canvas = document.getElementById('canvas');
                if (!canvas) return;
                const width = $1;
                const height = $2;
                if (canvas.width != width) canvas.width = width;
                if (canvas.height != height) canvas.height = height;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const bytesPerRow = $3;
                const src = HEAPU8.subarray($0, $0 + height * bytesPerRow);
                const imgData = ctx.createImageData(width, height);
                for (let y = 0; y < height; y++) {
                  const srcOffset = y * bytesPerRow;
                  const dstOffset = y * width * 4;
                  imgData.data.set(
                      src.subarray(srcOffset, srcOffset + width * 4),
                      dstOffset);
                }
                ctx.putImageData(imgData, 0, 0);
                window.__traceViewerV2FrameRendered = true;
              },
              mapped, width, height, bytes_per_row);
          // clang-format on
          readback_buffer.Unmap();
          readback_buffer.Destroy();
        });
  }
}

}  // namespace traceviewer
