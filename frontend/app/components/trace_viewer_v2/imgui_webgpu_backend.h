#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_IMGUI_WEBGPU_BACKEND_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_IMGUI_WEBGPU_BACKEND_H_
#include <webgpu/webgpu_cpp.h>

#include "third_party/dear_imgui/imgui.h"

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

IMGUI_IMPL_API bool ImGui_ImplWGPU_Init(
    const ImGui_ImplWGPU_InitInfo* init_info);
IMGUI_IMPL_API void ImGui_ImplWGPU_Shutdown();
IMGUI_IMPL_API void ImGui_ImplWGPU_NewFrame();
IMGUI_IMPL_API void ImGui_ImplWGPU_RenderDrawData(
    ImDrawData* draw_data, wgpu::RenderPassEncoder pass_encoder);

IMGUI_IMPL_API void ImGui_ImplWGPU_InvalidateDeviceObjects();
IMGUI_IMPL_API bool ImGui_ImplWGPU_CreateDeviceObjects();

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_IMGUI_WEBGPU_BACKEND_H_
