#include "frontend/app/components/trace_viewer_v2/imgui_webgpu_backend.h"

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "imgui.h"

#ifndef IMGUI_DISABLE
extern ImGuiID ImHashData(const void* data_p, size_t data_size, ImU32 seed = 0);

// Memory align macro to round up to the nearest multiple of an alignment.
#define MEMALIGN(_SIZE, _ALIGN) (((_SIZE) + ((_ALIGN) - 1)) & ~((_ALIGN) - 1))

struct ImGui_ImplWGPU_Texture {
  wgpu::Texture texture;
  wgpu::TextureView texture_view;
};

struct RenderResources {
  wgpu::Sampler sampler;
  wgpu::Buffer uniforms;
  wgpu::BindGroup common_bind_group;
  wgpu::BindGroupLayout image_bind_group_layout;
  wgpu::BindGroupLayout flame_batch_bind_group_layout;
  std::unordered_map<ImGuiID, wgpu::BindGroup> image_bind_groups;
};

struct alignas(256) FlameBatchUniforms {
  float visible_start_hi = 0.0f;
  float visible_start_lo = 0.0f;
  float px_per_us = 0.0f;
  float screen_x_offset = 0.0f;
  float timeline_right = 0.0f;
  float y_top = 0.0f;
  float y_bottom = 0.0f;
  float min_width_px = 1.0f;
  float padding_right_px = 0.5f;
  float alpha_multiplier = 1.0f;
  float pad[54] = {};
};
static_assert(sizeof(FlameBatchUniforms) == 256,
              "FlameBatchUniforms must be 256-byte aligned for dynamic offset");

struct FrameResources {
  wgpu::Buffer index_buffer;
  wgpu::Buffer vertex_buffer;
  std::vector<ImDrawIdx> index_buffer_host;
  std::vector<ImDrawVert> vertex_buffer_host;
  wgpu::Buffer flame_batch_uniform_buffer;
  wgpu::BindGroup flame_batch_bind_group;
  std::vector<FlameBatchUniforms> flame_batch_host;
};

// Shader uniform data
struct Uniforms {
  float mvp[4][4];
  float gamma;
};

// Main backend data structure.
struct ImGui_ImplWGPU_Data {
  ImGui_ImplWGPU_InitInfo init_info;
  wgpu::Device device;
  wgpu::Queue default_queue;
  wgpu::Queue queue;
  wgpu::RenderPipeline pipeline_state;
  wgpu::RenderPipeline flame_pipeline_state;
  wgpu::Buffer flame_instance_buffer;
  size_t flame_instance_capacity = 0;
  size_t flame_instance_count = 0;
  std::vector<ImGui_ImplWGPU_FlameBatchParams> pending_flame_batches;

  RenderResources render_resources;
  std::vector<FrameResources> frame_resources;
  uint32_t frame_index = UINT32_MAX;
};

static ImGui_ImplWGPU_Data* ImGui_ImplWGPU_GetBackendData() {
  return ImGui::GetCurrentContext()
             ? static_cast<ImGui_ImplWGPU_Data*>(
                   ImGui::GetIO().BackendRendererUserData)
             : nullptr;
}

static void FlameBatchCallback(const ImDrawList*, const ImDrawCmd*) {}

// TODO(nancyly): Move shaders to separate .wgsl files.
static const char kShaderVertWgsl[] = R"(
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct Uniforms {
    mvp: mat4x4<f32>,
    gamma: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}
)";
static const char kShaderFragWgsl[] = R"(
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct Uniforms {
    mvp: mat4x4<f32>,
    gamma: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(0) var t: texture_2d<f32>;
@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color * textureSample(t, s, in.uv);
    let corrected_color = pow(color.rgb, vec3<f32>(uniforms.gamma));
    return vec4<f32>(corrected_color, color.a);
}
)";

static const char kFlameShaderVertWgsl[] = R"(
struct InstanceInput {
    @location(0) start_hi: f32,
    @location(1) start_lo: f32,
    @location(2) duration: f32,
    @location(3) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
struct Uniforms {
    mvp: mat4x4<f32>,
    gamma: f32,
};
struct FlameBatchUniforms {
    visible_start_hi: f32,
    visible_start_lo: f32,
    px_per_us: f32,
    screen_x_offset: f32,
    timeline_right: f32,
    y_top: f32,
    y_bottom: f32,
    min_width_px: f32,
    padding_right_px: f32,
    alpha_multiplier: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> batch: FlameBatchUniforms;
@vertex
fn main(@builtin(vertex_index) vid: u32, in: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let dt = (in.start_hi - batch.visible_start_hi) + (in.start_lo - batch.visible_start_lo);
    let raw_left = batch.screen_x_offset + dt * batch.px_per_us;
    var raw_right = raw_left + in.duration * batch.px_per_us;
    raw_right = max(raw_right, raw_left + batch.min_width_px) - batch.padding_right_px;
    let left = clamp(raw_left, batch.screen_x_offset, batch.timeline_right);
    let right = clamp(raw_right, batch.screen_x_offset, batch.timeline_right);
    if (right <= left) {
        out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    var x = left;
    if (vid == 1u || vid == 2u || vid == 4u) {
        x = right;
    }
    var y = batch.y_top;
    if (vid == 2u || vid == 4u || vid == 5u) {
        y = batch.y_bottom;
    }
    out.position = uniforms.mvp * vec4<f32>(x, y, 0.0, 1.0);
    out.color = vec4<f32>(in.color.rgb, in.color.a * batch.alpha_multiplier);
    return out;
}
)";

static const char kFlameShaderFragWgsl[] = R"(
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
struct Uniforms {
    mvp: mat4x4<f32>,
    gamma: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let corrected_color = pow(in.color.rgb, vec3<f32>(uniforms.gamma));
    return vec4<f32>(corrected_color, in.color.a);
}
)";

static wgpu::ShaderModule CreateShaderModule(const char* wgsl_source) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  wgpu::ShaderModuleWGSLDescriptor wgsl_descriptor;
  wgsl_descriptor.code = wgsl_source;
  wgpu::ShaderModuleDescriptor module_descriptor{};
  module_descriptor.nextInChain = &wgsl_descriptor;
  return bd->device.CreateShaderModule(&module_descriptor);
}

static wgpu::BindGroup CreateImageBindGroup(wgpu::BindGroupLayout layout,
                                            wgpu::TextureView texture) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  wgpu::BindGroupEntry image_bg_entry{};
  image_bg_entry.binding = 0;
  image_bg_entry.textureView = texture;
  wgpu::BindGroupDescriptor image_bg_descriptor{};
  image_bg_descriptor.layout = layout;
  image_bg_descriptor.entryCount = 1;
  image_bg_descriptor.entries = &image_bg_entry;
  return bd->device.CreateBindGroup(&image_bg_descriptor);
}

static void SetupRenderState(ImDrawData* draw_data,
                             wgpu::RenderPassEncoder pass_encoder,
                             FrameResources* frame,
                             bool write_uniforms = true) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (write_uniforms) {
    float left = draw_data->DisplayPos.x;
    float right = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float top = draw_data->DisplayPos.y;
    float bottom = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    Uniforms u = {
        .mvp =
            {
                {2.0f / (right - left), 0.0f, 0.0f, 0.0f},
                {0.0f, 2.0f / (top - bottom), 0.0f, 0.0f},
                {0.0f, 0.0f, 0.5f, 0.0f},
                {(right + left) / (left - right),
                 (top + bottom) / (bottom - top), 0.5f, 1.0f},
            },
        .gamma = 1.0f,
    };
    switch (bd->init_info.target_format) {
      case wgpu::TextureFormat::BGRA8UnormSrgb:
      case wgpu::TextureFormat::RGBA8UnormSrgb:
        u.gamma = 2.2f;
        break;
      default:
        break;
    }
    bd->default_queue.WriteBuffer(bd->render_resources.uniforms, 0, &u,
                                  sizeof(Uniforms));
  }
  pass_encoder.SetViewport(
      0, 0, draw_data->FramebufferScale.x * draw_data->DisplaySize.x,
      draw_data->FramebufferScale.y * draw_data->DisplaySize.y, 0, 1);
  pass_encoder.SetVertexBuffer(
      0, frame->vertex_buffer, 0,
      frame->vertex_buffer_host.size() * sizeof(ImDrawVert));
  pass_encoder.SetIndexBuffer(
      frame->index_buffer,
      (sizeof(ImDrawIdx) == 2) ? wgpu::IndexFormat::Uint16
                               : wgpu::IndexFormat::Uint32,
      0, frame->index_buffer_host.size() * sizeof(ImDrawIdx));
  pass_encoder.SetPipeline(bd->pipeline_state);
  pass_encoder.SetBindGroup(0, bd->render_resources.common_bind_group);
  wgpu::Color blend_color{0.f, 0.f, 0.f, 0.f};
  pass_encoder.SetBlendConstant(&blend_color);
}

void ImGui_ImplWGPU_RenderDrawData(ImDrawData* draw_data,
                                   wgpu::RenderPassEncoder pass_encoder) {
  int fb_width = static_cast<int>(draw_data->DisplaySize.x *
                                  draw_data->FramebufferScale.x);
  int fb_height = static_cast<int>(draw_data->DisplaySize.y *
                                   draw_data->FramebufferScale.y);
  if (fb_width <= 0 || fb_height <= 0 || draw_data->CmdListsCount == 0) return;

  if (draw_data->Textures != nullptr) {
    for (ImTextureData* tex : *draw_data->Textures) {
      if (tex->Status != ImTextureStatus_OK) {
        ImGui_ImplWGPU_UpdateTexture(tex);
      }
    }
  }

  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  bd->frame_index = (bd->frame_index + 1) % bd->init_info.num_frames_in_flight;
  FrameResources* frame = &bd->frame_resources[bd->frame_index];

  if (!frame->vertex_buffer ||
      frame->vertex_buffer_host.size() <
          static_cast<size_t>(draw_data->TotalVtxCount)) {
    frame->vertex_buffer_host.resize(draw_data->TotalVtxCount * 2);
    wgpu::BufferDescriptor vb_desc{};
    vb_desc.label = "Dear ImGui Vertex buffer";
    vb_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Vertex;
    vb_desc.size = frame->vertex_buffer_host.size() * sizeof(ImDrawVert);
    frame->vertex_buffer = bd->device.CreateBuffer(&vb_desc);
    CHECK(frame->vertex_buffer) << "Failed to create vertex buffer.";
  }
  if (!frame->index_buffer ||
      frame->index_buffer_host.size() <
          static_cast<size_t>(draw_data->TotalIdxCount)) {
    frame->index_buffer_host.resize(draw_data->TotalIdxCount * 2);
    wgpu::BufferDescriptor ib_desc{};
    ib_desc.label = "Dear ImGui Index buffer";
    ib_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index;
    ib_desc.size = frame->index_buffer_host.size() * sizeof(ImDrawIdx);
    frame->index_buffer = bd->device.CreateBuffer(&ib_desc);
    CHECK(frame->index_buffer) << "Failed to create index buffer.";
  }

  ImDrawVert* vtx_destination = frame->vertex_buffer_host.data();
  ImDrawIdx* idx_destination = frame->index_buffer_host.data();
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    memcpy(vtx_destination, cmd_list->VtxBuffer.Data,
           cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    memcpy(idx_destination, cmd_list->IdxBuffer.Data,
           cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx_destination += cmd_list->VtxBuffer.Size;
    idx_destination += cmd_list->IdxBuffer.Size;
  }
  uint64_t vb_write_size =
      (char*)vtx_destination - (char*)frame->vertex_buffer_host.data();
  uint64_t ib_write_size =
      (char*)idx_destination - (char*)frame->index_buffer_host.data();

  // WebGPU requires WriteBuffer size to be a multiple of 4.
  vb_write_size = MEMALIGN(vb_write_size, 4);
  ib_write_size = MEMALIGN(ib_write_size, 4);

  bd->default_queue.WriteBuffer(
      frame->vertex_buffer, 0, frame->vertex_buffer_host.data(), vb_write_size);
  bd->default_queue.WriteBuffer(frame->index_buffer, 0,
                                frame->index_buffer_host.data(), ib_write_size);

  if (!bd->pending_flame_batches.empty() && bd->flame_pipeline_state &&
      bd->flame_instance_buffer &&
      bd->render_resources.flame_batch_bind_group_layout) {
    const size_t num_batches = bd->pending_flame_batches.size();
    if (!frame->flame_batch_uniform_buffer ||
        frame->flame_batch_host.size() < num_batches) {
      const size_t new_cap = std::max<size_t>(num_batches + 64, 256);
      frame->flame_batch_host.resize(new_cap);
      wgpu::BufferDescriptor batch_ub_desc{};
      batch_ub_desc.label = "Dear ImGui Flame Batch Uniform buffer";
      batch_ub_desc.usage =
          wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform;
      batch_ub_desc.size = new_cap * sizeof(FlameBatchUniforms);
      frame->flame_batch_uniform_buffer =
          bd->device.CreateBuffer(&batch_ub_desc);
      CHECK(frame->flame_batch_uniform_buffer)
          << "Failed to create flame batch uniform buffer.";

      wgpu::BindGroupEntry batch_bg_entry{};
      batch_bg_entry.binding = 0;
      batch_bg_entry.buffer = frame->flame_batch_uniform_buffer;
      batch_bg_entry.offset = 0;
      batch_bg_entry.size = sizeof(FlameBatchUniforms);
      wgpu::BindGroupDescriptor batch_bg_desc{};
      batch_bg_desc.layout = bd->render_resources.flame_batch_bind_group_layout;
      batch_bg_desc.entryCount = 1;
      batch_bg_desc.entries = &batch_bg_entry;
      frame->flame_batch_bind_group =
          bd->device.CreateBindGroup(&batch_bg_desc);
    }
    for (size_t i = 0; i < num_batches; ++i) {
      const auto& src = bd->pending_flame_batches[i];
      FlameBatchUniforms& dst = frame->flame_batch_host[i];
      dst.visible_start_hi = static_cast<float>(src.visible_start_us);
      dst.visible_start_lo = static_cast<float>(
          src.visible_start_us - static_cast<double>(dst.visible_start_hi));
      dst.px_per_us = src.px_per_us;
      dst.screen_x_offset = src.screen_x_offset;
      dst.timeline_right = src.screen_x_offset + src.timeline_width;
      dst.y_top = src.y_top;
      dst.y_bottom = src.y_bottom;
      dst.min_width_px = src.min_width_px;
      dst.padding_right_px = src.padding_right_px;
      dst.alpha_multiplier = src.alpha_multiplier;
    }
    bd->default_queue.WriteBuffer(frame->flame_batch_uniform_buffer, 0,
                                  frame->flame_batch_host.data(),
                                  num_batches * sizeof(FlameBatchUniforms));
  }

  SetupRenderState(draw_data, pass_encoder, frame, /*write_uniforms=*/true);

  int global_vtx_offset = 0;
  int global_idx_offset = 0;
  ImVec2 clip_off = draw_data->DisplayPos;
  ImVec2 clip_scale = draw_data->FramebufferScale;
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList* cmd_list = draw_data->CmdLists[n];
    for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
      const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
      if (pcmd->UserCallback) {
        if (pcmd->UserCallback == ImDrawCallback_ResetRenderState) {
          SetupRenderState(draw_data, pass_encoder, frame,
                           /*write_uniforms=*/false);
        } else if (pcmd->UserCallback == FlameBatchCallback) {
          const size_t batch_idx =
              reinterpret_cast<uintptr_t>(pcmd->UserCallbackData);
          if (batch_idx < bd->pending_flame_batches.size() &&
              bd->flame_pipeline_state && bd->flame_instance_buffer &&
              frame->flame_batch_bind_group) {
            const auto& batch = bd->pending_flame_batches[batch_idx];
            if (batch.instance_count > 0 &&
                static_cast<uint64_t>(batch.first_instance) +
                        batch.instance_count <=
                    bd->flame_instance_count) {
              ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x,
                              (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
              ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x,
                              (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);
              if (clip_min.x < 0.0f) clip_min.x = 0.0f;
              if (clip_min.y < 0.0f) clip_min.y = 0.0f;
              if (clip_max.x > fb_width) {
                clip_max.x = static_cast<float>(fb_width);
              }
              if (clip_max.y > fb_height) {
                clip_max.y = static_cast<float>(fb_height);
              }
              if (clip_max.x > clip_min.x && clip_max.y > clip_min.y) {
                pass_encoder.SetScissorRect(
                    static_cast<uint32_t>(clip_min.x),
                    static_cast<uint32_t>(clip_min.y),
                    static_cast<uint32_t>(clip_max.x - clip_min.x),
                    static_cast<uint32_t>(clip_max.y - clip_min.y));
                pass_encoder.SetPipeline(bd->flame_pipeline_state);
                pass_encoder.SetBindGroup(
                    0, bd->render_resources.common_bind_group);
                uint32_t dynamic_offset = static_cast<uint32_t>(
                    batch_idx * sizeof(FlameBatchUniforms));
                pass_encoder.SetBindGroup(1, frame->flame_batch_bind_group, 1,
                                          &dynamic_offset);
                pass_encoder.SetVertexBuffer(
                    0, bd->flame_instance_buffer,
                    static_cast<uint64_t>(batch.first_instance) *
                        sizeof(ImGui_ImplWGPU_FlameInstance),
                    static_cast<uint64_t>(batch.instance_count) *
                        sizeof(ImGui_ImplWGPU_FlameInstance));
                pass_encoder.Draw(6, batch.instance_count, 0, 0);
              }
            }
          }
        } else {
          pcmd->UserCallback(cmd_list, pcmd);
        }
      } else {
        ImTextureID tex_id = pcmd->GetTexID();
        ImGuiID tex_id_hash = ImHashData(&tex_id, sizeof(tex_id));
        auto it = bd->render_resources.image_bind_groups.find(tex_id_hash);
        if (it != bd->render_resources.image_bind_groups.end()) {
          pass_encoder.SetBindGroup(1, it->second);
        } else {
          wgpu::TextureView texture_view = wgpu::TextureView::Acquire(
              reinterpret_cast<WGPUTextureView>(tex_id));
          wgpu::BindGroup image_bind_group = CreateImageBindGroup(
              bd->render_resources.image_bind_group_layout, texture_view);
          bd->render_resources.image_bind_groups[tex_id_hash] =
              image_bind_group;
          pass_encoder.SetBindGroup(1, image_bind_group);
          texture_view.MoveToCHandle();
        }

        ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x,
                        (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
        ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x,
                        (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);

        if (clip_min.x < 0.0f) clip_min.x = 0.0f;
        if (clip_min.y < 0.0f) clip_min.y = 0.0f;
        if (clip_max.x > fb_width) clip_max.x = static_cast<float>(fb_width);
        if (clip_max.y > fb_height) clip_max.y = static_cast<float>(fb_height);
        if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) continue;

        pass_encoder.SetScissorRect(
            static_cast<uint32_t>(clip_min.x),
            static_cast<uint32_t>(clip_min.y),
            static_cast<uint32_t>(clip_max.x - clip_min.x),
            static_cast<uint32_t>(clip_max.y - clip_min.y));

        pass_encoder.DrawIndexed(pcmd->ElemCount, 1,
                                 pcmd->IdxOffset + global_idx_offset,
                                 pcmd->VtxOffset + global_vtx_offset, 0);
      }
    }
    global_idx_offset += cmd_list->IdxBuffer.Size;
    global_vtx_offset += cmd_list->VtxBuffer.Size;
  }
  bd->pending_flame_batches.clear();
}

void ImGui_ImplWGPU_DestroyTexture(ImTextureData* tex) {
  if (auto* backend_tex =
          static_cast<ImGui_ImplWGPU_Texture*>(tex->BackendUserData)) {
    ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
    if (bd) {
      ImTextureID tex_id = tex->TexID;
      ImGuiID tex_id_hash = ImHashData(&tex_id, sizeof(tex_id));
      bd->render_resources.image_bind_groups.erase(tex_id_hash);
    }
    backend_tex->texture_view = nullptr;
    backend_tex->texture = nullptr;
    delete backend_tex;

    tex->SetTexID(ImTextureID_Invalid);
    tex->BackendUserData = nullptr;
  }
  tex->SetStatus(ImTextureStatus_Destroyed);
}

void ImGui_ImplWGPU_UpdateTexture(ImTextureData* tex) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd || !bd->device) return;

  if (tex->Status == ImTextureStatus_WantCreate) {
    CHECK_EQ(tex->TexID, ImTextureID_Invalid);
    CHECK_EQ(tex->BackendUserData, nullptr);
    CHECK_EQ(tex->Format, ImTextureFormat_RGBA32);
    auto* backend_tex = new ImGui_ImplWGPU_Texture();

    wgpu::TextureDescriptor tex_desc{};
    tex_desc.label = "Dear ImGui Texture";
    tex_desc.dimension = wgpu::TextureDimension::e2D;
    tex_desc.size = {static_cast<uint32_t>(tex->Width),
                     static_cast<uint32_t>(tex->Height), 1};
    tex_desc.sampleCount = 1;
    tex_desc.format = wgpu::TextureFormat::RGBA8Unorm;
    tex_desc.mipLevelCount = 1;
    tex_desc.usage =
        wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
    backend_tex->texture = bd->device.CreateTexture(&tex_desc);

    wgpu::TextureViewDescriptor tex_view_desc{};
    tex_view_desc.format = wgpu::TextureFormat::RGBA8Unorm;
    tex_view_desc.dimension = wgpu::TextureViewDimension::e2D;
    tex_view_desc.baseMipLevel = 0;
    tex_view_desc.mipLevelCount = 1;
    tex_view_desc.baseArrayLayer = 0;
    tex_view_desc.arrayLayerCount = 1;
    tex_view_desc.aspect = wgpu::TextureAspect::All;
    backend_tex->texture_view = backend_tex->texture.CreateView(&tex_view_desc);

    tex->SetTexID(
        reinterpret_cast<ImTextureID>(backend_tex->texture_view.Get()));
    tex->BackendUserData = backend_tex;
  }

  if (tex->Status == ImTextureStatus_WantCreate ||
      tex->Status == ImTextureStatus_WantUpdates) {
    auto* backend_tex =
        static_cast<ImGui_ImplWGPU_Texture*>(tex->BackendUserData);
    CHECK_EQ(tex->Format, ImTextureFormat_RGBA32);

    int upload_x = 0;
    int upload_y = 0;
    int upload_w = tex->Width;
    int upload_h = tex->Height;

    if (tex->Status == ImTextureStatus_WantUpdates &&
        tex->UpdateRect.w > 0 && tex->UpdateRect.h > 0) {
      upload_x = tex->UpdateRect.x;
      upload_y = tex->UpdateRect.y;
      upload_w = tex->UpdateRect.w;
      upload_h = tex->UpdateRect.h;
    }

    wgpu::ImageCopyTexture destination = {};
    destination.texture = backend_tex->texture;
    destination.origin = {static_cast<uint32_t>(upload_x),
                          static_cast<uint32_t>(upload_y), 0};
    destination.aspect = wgpu::TextureAspect::All;

    wgpu::TextureDataLayout layout = {};
    layout.offset = static_cast<uint64_t>(upload_y * tex->Width + upload_x) * 4;
    layout.bytesPerRow = static_cast<uint32_t>(tex->Width * 4);
    layout.rowsPerImage = static_cast<uint32_t>(tex->Height);

    wgpu::Extent3D copy_size = {static_cast<uint32_t>(upload_w),
                                static_cast<uint32_t>(upload_h), 1};
    bd->queue.WriteTexture(&destination, tex->Pixels,
                           static_cast<size_t>(tex->Width * tex->Height * 4),
                           &layout, &copy_size);
    tex->SetStatus(ImTextureStatus_OK);
  }

  if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
    ImGui_ImplWGPU_DestroyTexture(tex);
  }
}

static void CreateUniformBuffer() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  wgpu::BufferDescriptor ub_desc{};
  ub_desc.label = "Dear ImGui uniform buffer";
  ub_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform;
  ub_desc.size = MEMALIGN(sizeof(Uniforms), 16);
  bd->render_resources.uniforms = bd->device.CreateBuffer(&ub_desc);
}

bool ImGui_ImplWGPU_CreateDeviceObjects() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd || !bd->device) {
    LOG(WARNING)
        << "CreateDeviceObjects called with null backend data or device.";
    return false;
  }
  if (bd->pipeline_state) ImGui_ImplWGPU_InvalidateDeviceObjects();

  wgpu::BindGroupLayoutEntry common_bgl_entries[2];
  common_bgl_entries[0].binding = 0;
  common_bgl_entries[0].visibility =
      wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
  common_bgl_entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
  common_bgl_entries[1].binding = 1;
  common_bgl_entries[1].visibility = wgpu::ShaderStage::Fragment;
  common_bgl_entries[1].sampler.type = wgpu::SamplerBindingType::Filtering;
  wgpu::BindGroupLayoutDescriptor common_bgl_desc{};
  common_bgl_desc.entryCount =
      sizeof(common_bgl_entries) / sizeof(common_bgl_entries[0]);
  common_bgl_desc.entries = common_bgl_entries;
  wgpu::BindGroupLayout common_bgl =
      bd->device.CreateBindGroupLayout(&common_bgl_desc);

  wgpu::BindGroupLayoutEntry image_bgl_entry{};
  image_bgl_entry.binding = 0;
  image_bgl_entry.visibility = wgpu::ShaderStage::Fragment;
  image_bgl_entry.texture.sampleType = wgpu::TextureSampleType::Float;
  image_bgl_entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
  wgpu::BindGroupLayoutDescriptor image_bgl_desc{};
  image_bgl_desc.entryCount = 1;
  image_bgl_desc.entries = &image_bgl_entry;
  bd->render_resources.image_bind_group_layout =
      bd->device.CreateBindGroupLayout(&image_bgl_desc);

  std::vector<wgpu::BindGroupLayout> bg_layouts = {
      common_bgl, bd->render_resources.image_bind_group_layout};
  wgpu::PipelineLayoutDescriptor layout_desc{};
  layout_desc.bindGroupLayoutCount = static_cast<uint32_t>(bg_layouts.size());
  layout_desc.bindGroupLayouts = bg_layouts.data();
  wgpu::PipelineLayout pipeline_layout =
      bd->device.CreatePipelineLayout(&layout_desc);

  wgpu::ShaderModule vertex_shader_module = CreateShaderModule(kShaderVertWgsl);
  wgpu::ShaderModule fragment_shader_module =
      CreateShaderModule(kShaderFragWgsl);

  wgpu::VertexAttribute attribute_descriptors[3];
  attribute_descriptors[0].format = wgpu::VertexFormat::Float32x2;
  attribute_descriptors[0].offset =
      static_cast<uint64_t>(offsetof(ImDrawVert, pos));
  attribute_descriptors[0].shaderLocation = 0;
  attribute_descriptors[1].format = wgpu::VertexFormat::Float32x2;
  attribute_descriptors[1].offset =
      static_cast<uint64_t>(offsetof(ImDrawVert, uv));
  attribute_descriptors[1].shaderLocation = 1;
  attribute_descriptors[2].format = wgpu::VertexFormat::Unorm8x4;
  attribute_descriptors[2].offset =
      static_cast<uint64_t>(offsetof(ImDrawVert, col));
  attribute_descriptors[2].shaderLocation = 2;

  wgpu::VertexBufferLayout buffer_layout{};
  buffer_layout.arrayStride = sizeof(ImDrawVert);
  buffer_layout.attributeCount = 3;
  buffer_layout.attributes = attribute_descriptors;

  wgpu::VertexState vertex_state{};
  vertex_state.module = vertex_shader_module;
  vertex_state.entryPoint = "main";
  vertex_state.bufferCount = 1;
  vertex_state.buffers = &buffer_layout;

  wgpu::BlendState blend_state{};
  blend_state.color = {wgpu::BlendOperation::Add, wgpu::BlendFactor::SrcAlpha,
                       wgpu::BlendFactor::OneMinusSrcAlpha};
  blend_state.alpha = {wgpu::BlendOperation::Add, wgpu::BlendFactor::One,
                       wgpu::BlendFactor::OneMinusSrcAlpha};
  wgpu::ColorTargetState color_target_state{};
  color_target_state.format = bd->init_info.target_format;
  color_target_state.blend = &blend_state;
  color_target_state.writeMask = wgpu::ColorWriteMask::All;

  wgpu::FragmentState fragment_state{};
  fragment_state.module = fragment_shader_module;
  fragment_state.entryPoint = "main";
  fragment_state.targetCount = 1;
  fragment_state.targets = &color_target_state;

  wgpu::DepthStencilState depth_stencil_state{};
  depth_stencil_state.format = bd->init_info.depth_stencil_format;
  depth_stencil_state.depthWriteEnabled = false;
  depth_stencil_state.depthCompare = wgpu::CompareFunction::Always;

  wgpu::RenderPipelineDescriptor desc{};
  desc.layout = pipeline_layout;
  desc.vertex = vertex_state;
  desc.fragment = &fragment_state;
  desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  desc.multisample = bd->init_info.multisample_state;
  if (bd->init_info.depth_stencil_format != wgpu::TextureFormat::Undefined) {
    desc.depthStencil = &depth_stencil_state;
  }

  bd->pipeline_state = bd->device.CreateRenderPipeline(&desc);
  CHECK(bd->pipeline_state) << "Failed to create ImGui render pipeline.";

  wgpu::BindGroupLayoutEntry flame_batch_bgl_entry{};
  flame_batch_bgl_entry.binding = 0;
  flame_batch_bgl_entry.visibility = wgpu::ShaderStage::Vertex;
  flame_batch_bgl_entry.buffer.type = wgpu::BufferBindingType::Uniform;
  flame_batch_bgl_entry.buffer.hasDynamicOffset = true;
  flame_batch_bgl_entry.buffer.minBindingSize = sizeof(FlameBatchUniforms);
  wgpu::BindGroupLayoutDescriptor flame_batch_bgl_desc{};
  flame_batch_bgl_desc.entryCount = 1;
  flame_batch_bgl_desc.entries = &flame_batch_bgl_entry;
  bd->render_resources.flame_batch_bind_group_layout =
      bd->device.CreateBindGroupLayout(&flame_batch_bgl_desc);

  std::vector<wgpu::BindGroupLayout> flame_bg_layouts = {
      common_bgl, bd->render_resources.flame_batch_bind_group_layout};
  wgpu::PipelineLayoutDescriptor flame_layout_desc{};
  flame_layout_desc.bindGroupLayoutCount =
      static_cast<uint32_t>(flame_bg_layouts.size());
  flame_layout_desc.bindGroupLayouts = flame_bg_layouts.data();
  wgpu::PipelineLayout flame_pipeline_layout =
      bd->device.CreatePipelineLayout(&flame_layout_desc);

  wgpu::ShaderModule flame_vs_module = CreateShaderModule(kFlameShaderVertWgsl);
  wgpu::ShaderModule flame_fs_module = CreateShaderModule(kFlameShaderFragWgsl);

  wgpu::VertexAttribute flame_attrs[4];
  flame_attrs[0].format = wgpu::VertexFormat::Float32;
  flame_attrs[0].offset =
      static_cast<uint64_t>(offsetof(ImGui_ImplWGPU_FlameInstance, start_hi));
  flame_attrs[0].shaderLocation = 0;
  flame_attrs[1].format = wgpu::VertexFormat::Float32;
  flame_attrs[1].offset =
      static_cast<uint64_t>(offsetof(ImGui_ImplWGPU_FlameInstance, start_lo));
  flame_attrs[1].shaderLocation = 1;
  flame_attrs[2].format = wgpu::VertexFormat::Float32;
  flame_attrs[2].offset =
      static_cast<uint64_t>(offsetof(ImGui_ImplWGPU_FlameInstance, duration));
  flame_attrs[2].shaderLocation = 2;
  flame_attrs[3].format = wgpu::VertexFormat::Unorm8x4;
  flame_attrs[3].offset =
      static_cast<uint64_t>(offsetof(ImGui_ImplWGPU_FlameInstance, color));
  flame_attrs[3].shaderLocation = 3;

  wgpu::VertexBufferLayout flame_vb_layout{};
  flame_vb_layout.arrayStride = sizeof(ImGui_ImplWGPU_FlameInstance);
  flame_vb_layout.stepMode = wgpu::VertexStepMode::Instance;
  flame_vb_layout.attributeCount = 4;
  flame_vb_layout.attributes = flame_attrs;

  wgpu::VertexState flame_vertex_state{};
  flame_vertex_state.module = flame_vs_module;
  flame_vertex_state.entryPoint = "main";
  flame_vertex_state.bufferCount = 1;
  flame_vertex_state.buffers = &flame_vb_layout;

  wgpu::FragmentState flame_fragment_state{};
  flame_fragment_state.module = flame_fs_module;
  flame_fragment_state.entryPoint = "main";
  flame_fragment_state.targetCount = 1;
  flame_fragment_state.targets = &color_target_state;

  wgpu::RenderPipelineDescriptor flame_desc{};
  flame_desc.layout = flame_pipeline_layout;
  flame_desc.vertex = flame_vertex_state;
  flame_desc.fragment = &flame_fragment_state;
  flame_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  flame_desc.multisample = bd->init_info.multisample_state;
  if (bd->init_info.depth_stencil_format != wgpu::TextureFormat::Undefined) {
    flame_desc.depthStencil = &depth_stencil_state;
  }

  bd->flame_pipeline_state = bd->device.CreateRenderPipeline(&flame_desc);
  CHECK(bd->flame_pipeline_state)
      << "Failed to create ImGui flame instance render pipeline.";

  wgpu::SamplerDescriptor sampler_desc = {};
  sampler_desc.minFilter = wgpu::FilterMode::Linear;
  sampler_desc.magFilter = wgpu::FilterMode::Linear;
  bd->render_resources.sampler = bd->device.CreateSampler(&sampler_desc);

  CreateUniformBuffer();

  wgpu::BindGroupEntry common_bg_entries[2];
  common_bg_entries[0].binding = 0;
  common_bg_entries[0].buffer = bd->render_resources.uniforms;
  common_bg_entries[0].size = MEMALIGN(sizeof(Uniforms), 16);
  common_bg_entries[1].binding = 1;
  common_bg_entries[1].sampler = bd->render_resources.sampler;

  wgpu::BindGroupDescriptor common_bg_desc{};
  common_bg_desc.layout = common_bgl;
  common_bg_desc.entryCount =
      sizeof(common_bg_entries) / sizeof(common_bg_entries[0]);
  common_bg_desc.entries = common_bg_entries;
  bd->render_resources.common_bind_group =
      bd->device.CreateBindGroup(&common_bg_desc);

  return true;
}

void ImGui_ImplWGPU_InvalidateDeviceObjects() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd || !bd->device) return;

  bd->pipeline_state = nullptr;
  bd->flame_pipeline_state = nullptr;
  bd->flame_instance_buffer = nullptr;
  bd->flame_instance_capacity = 0;
  bd->flame_instance_count = 0;
  bd->pending_flame_batches.clear();
  bd->render_resources = {};

  for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
    if (tex->RefCount == 1) {
      ImGui_ImplWGPU_DestroyTexture(tex);
    }
  }

  for (auto& frame : bd->frame_resources) {
    frame = {};
  }
}

bool ImGui_ImplWGPU_Init(const ImGui_ImplWGPU_InitInfo* init_info) {
  ImGuiIO& io = ImGui::GetIO();
  IMGUI_CHECKVERSION();
  CHECK_EQ(io.BackendRendererUserData, nullptr)
      << "Already initialized a renderer backend!";

  ImGui_ImplWGPU_Data* bd = new ImGui_ImplWGPU_Data();
  io.BackendRendererUserData = static_cast<void*>(bd);
  io.BackendRendererName = "imgui_impl_wgpu_cpp";
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;

  bd->init_info = *init_info;
  bd->device = init_info->device;
  bd->default_queue = init_info->device.GetQueue();
  bd->queue = bd->default_queue;
  bd->frame_resources.resize(init_info->num_frames_in_flight);
  bd->frame_index = UINT32_MAX;

  LOG(INFO) << absl::StrFormat(
      "ImGui_ImplWGPU_Init: Succeeded with %d frames in flight.",
      init_info->num_frames_in_flight);
  return true;
}

void ImGui_ImplWGPU_Shutdown() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  CHECK_NE(bd, nullptr)
      << "No renderer backend to shutdown, or already shutdown?";
  ImGuiIO& io = ImGui::GetIO();
  ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();

  ImGui_ImplWGPU_InvalidateDeviceObjects();
  bd->frame_resources.clear();
  io.BackendRendererName = nullptr;
  io.BackendRendererUserData = nullptr;
  io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset |
                       ImGuiBackendFlags_RendererHasTextures);
  platform_io.ClearRendererHandlers();
  delete bd;

  LOG(INFO) << "ImGui_ImplWGPU_Shutdown: Succeeded.";
}

void ImGui_ImplWGPU_NewFrame() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd->pipeline_state) {
    ImGui_ImplWGPU_CreateDeviceObjects();
  }
  bd->pending_flame_batches.clear();
}

bool ImGui_ImplWGPU_HasFlameInstanceBuffer() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  return bd != nullptr && bd->device && bd->flame_instance_buffer &&
         bd->flame_instance_count > 0;
}

void ImGui_ImplWGPU_UploadFlameInstances(
    const ImGui_ImplWGPU_FlameInstance* instances, size_t count) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd || !bd->device) return;
  if (count == 0 || instances == nullptr) {
    bd->flame_instance_count = 0;
    return;
  }
  if (!bd->flame_instance_buffer || bd->flame_instance_capacity < count) {
    const size_t new_cap = std::max(count, bd->flame_instance_capacity * 3 / 2);
    wgpu::BufferDescriptor desc{};
    desc.label = "Dear ImGui Flame Instance buffer";
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Vertex;
    desc.size = new_cap * sizeof(ImGui_ImplWGPU_FlameInstance);
    bd->flame_instance_buffer = bd->device.CreateBuffer(&desc);
    if (!bd->flame_instance_buffer) {
      bd->flame_instance_capacity = 0;
      bd->flame_instance_count = 0;
      return;
    }
    bd->flame_instance_capacity = new_cap;
  }
  bd->default_queue.WriteBuffer(bd->flame_instance_buffer, 0, instances,
                                count * sizeof(ImGui_ImplWGPU_FlameInstance));
  bd->flame_instance_count = count;
}

void ImGui_ImplWGPU_AddFlameBatch(
    ImDrawList* draw_list, const ImGui_ImplWGPU_FlameBatchParams& params) {
  if (!draw_list || params.instance_count == 0) return;
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd || !bd->flame_instance_buffer ||
      static_cast<uint64_t>(params.first_instance) + params.instance_count >
          bd->flame_instance_count) {
    return;
  }
  const size_t batch_idx = bd->pending_flame_batches.size();
  bd->pending_flame_batches.push_back(params);
  draw_list->AddCallback(FlameBatchCallback,
                         reinterpret_cast<void*>(batch_idx));
  draw_list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

#endif  // #ifndef IMGUI_DISABLE
