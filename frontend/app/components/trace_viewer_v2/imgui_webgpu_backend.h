#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_IMGUI_WEBGPU_BACKEND_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_IMGUI_WEBGPU_BACKEND_H_
#include <cstddef>
#include <cstdint>

#include "imgui.h"
#include "webgpu/webgpu_cpp.h"

struct ImGui_ImplWGPU_InitInfo {
  wgpu::Device device;
  int num_frames_in_flight = 3;
  wgpu::TextureFormat target_format = wgpu::TextureFormat::Undefined;
  wgpu::TextureFormat depth_stencil_format = wgpu::TextureFormat::Undefined;
  wgpu::MultisampleState multisample_state{};

  ImGui_ImplWGPU_InitInfo() {
    multisample_state.count = 1;
    multisample_state.mask = 0xFFFFFFFF;
    multisample_state.alphaToCoverageEnabled = false;
  }
};

struct ImGui_ImplWGPU_FlameInstance {
  float start_hi = 0.0f;
  float start_lo = 0.0f;
  float duration = 0.0f;
  uint32_t color = 0;
};

struct ImGui_ImplWGPU_FlameBatchParams {
  uint32_t first_instance = 0;
  uint32_t instance_count = 0;
  double visible_start_us = 0.0;
  float px_per_us = 0.0f;
  float screen_x_offset = 0.0f;
  float timeline_width = 0.0f;
  float y_top = 0.0f;
  float y_bottom = 0.0f;
  float min_width_px = 1.0f;
  float padding_right_px = 0.5f;
  float alpha_multiplier = 1.0f;
};

IMGUI_IMPL_API bool ImGui_ImplWGPU_Init(
    const ImGui_ImplWGPU_InitInfo* init_info);
IMGUI_IMPL_API void ImGui_ImplWGPU_Shutdown();
IMGUI_IMPL_API void ImGui_ImplWGPU_NewFrame();
IMGUI_IMPL_API void ImGui_ImplWGPU_RenderDrawData(
    ImDrawData* draw_data, wgpu::RenderPassEncoder pass_encoder);

IMGUI_IMPL_API void ImGui_ImplWGPU_InvalidateDeviceObjects();
IMGUI_IMPL_API bool ImGui_ImplWGPU_CreateDeviceObjects();

IMGUI_IMPL_API void ImGui_ImplWGPU_UpdateTexture(ImTextureData* tex);
IMGUI_IMPL_API void ImGui_ImplWGPU_DestroyTexture(ImTextureData* tex);

IMGUI_IMPL_API bool ImGui_ImplWGPU_HasFlameInstanceBuffer();
IMGUI_IMPL_API void ImGui_ImplWGPU_SetHeadlessFlameBatchMode(bool enabled);
IMGUI_IMPL_API void ImGui_ImplWGPU_UploadFlameInstances(
    const ImGui_ImplWGPU_FlameInstance* instances, size_t count);
IMGUI_IMPL_API void ImGui_ImplWGPU_AddFlameBatch(
    ImDrawList* draw_list, const ImGui_ImplWGPU_FlameBatchParams& params);

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_IMGUI_WEBGPU_BACKEND_H_
