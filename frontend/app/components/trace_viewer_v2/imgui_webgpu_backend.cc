#include "xprof/frontend/app/components/trace_viewer_v2/imgui_webgpu_backend.h"

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
#include "third_party/dear_imgui/imgui.h"

#ifndef IMGUI_DISABLE
extern ImGuiID ImHashData(const void* data_p, size_t data_size, ImU32 seed = 0);

// Memory align macro to round up to the nearest multiple of an alignment.
#define MEMALIGN(_SIZE, _ALIGN) (((_SIZE) + ((_ALIGN) - 1)) & ~((_ALIGN) - 1))

struct RenderResources {
  wgpu::Texture font_texture;
  wgpu::TextureView font_texture_view;
  wgpu::Sampler sampler;
  wgpu::Buffer uniforms;
  wgpu::BindGroup common_bind_group;
  wgpu::BindGroupLayout image_bind_group_layout;
  std::unordered_map<ImGuiID, wgpu::BindGroup> image_bind_groups;
};

struct FrameResources {
  wgpu::Buffer index_buffer;
  wgpu::Buffer vertex_buffer;
  std::vector<ImDrawIdx> index_buffer_host;
  std::vector<ImDrawVert> vertex_buffer_host;
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
  wgpu::RenderPipeline pipeline_state;

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

static wgpu::ShaderModule CreateShaderModule(const char* wgsl_source) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  wgpu::ShaderSourceWGSL wgsl_descriptor;
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
                             FrameResources* frame) {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  {
    float left = draw_data->DisplayPos.x;
    float right = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float top = draw_data->DisplayPos.y;
    float bottom = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    const float mvp[4][4] = {
        {2.0f / (right - left), 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f / (top - bottom), 0.0f, 0.0f},
        {0.0f, 0.0f, 0.5f, 0.0f},
        {(right + left) / (left - right), (top + bottom) / (bottom - top), 0.5f,
         1.0f},
    };
    bd->default_queue.WriteBuffer(bd->render_resources.uniforms,
                                  offsetof(Uniforms, mvp), &mvp, sizeof(mvp));
    float gamma = 1.0f;
    switch (bd->init_info.target_format) {
      case wgpu::TextureFormat::BGRA8UnormSrgb:
      case wgpu::TextureFormat::RGBA8UnormSrgb:
        gamma = 2.2f;
        break;
      default:
        break;
    }
    bd->default_queue.WriteBuffer(bd->render_resources.uniforms,
                                  offsetof(Uniforms, gamma), &gamma,
                                  sizeof(gamma));
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
  bd->default_queue.WriteBuffer(
      frame->vertex_buffer, 0, frame->vertex_buffer_host.data(), vb_write_size);
  bd->default_queue.WriteBuffer(frame->index_buffer, 0,
                                frame->index_buffer_host.data(), ib_write_size);

  SetupRenderState(draw_data, pass_encoder, frame);

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
          SetupRenderState(draw_data, pass_encoder, frame);
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
}

static void CreateFontsTexture() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  ImGuiIO& io = ImGui::GetIO();
  unsigned char* pixels;
  int i_width, i_height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &i_width, &i_height);
  uint32_t width = static_cast<uint32_t>(i_width);
  uint32_t height = static_cast<uint32_t>(i_height);

  wgpu::TextureDescriptor tex_desc{};
  tex_desc.label = "Dear ImGui Font Texture";
  tex_desc.size = {width, height, 1};
  tex_desc.sampleCount = 1;
  tex_desc.format = wgpu::TextureFormat::RGBA8Unorm;
  tex_desc.usage =
      wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
  bd->render_resources.font_texture = bd->device.CreateTexture(&tex_desc);

  wgpu::TextureViewDescriptor tex_view_desc{};
  tex_view_desc.dimension = wgpu::TextureViewDimension::e2D;
  tex_view_desc.aspect = wgpu::TextureAspect::All;
  bd->render_resources.font_texture_view =
      bd->render_resources.font_texture.CreateView(&tex_view_desc);

  wgpu::TexelCopyTextureInfo dst_view = {};
  dst_view.texture = bd->render_resources.font_texture;
  wgpu::TexelCopyBufferLayout layout = {};
  layout.bytesPerRow = width * 4;
  layout.rowsPerImage = height;
  wgpu::Extent3D size = {width, height, 1};
  bd->default_queue.WriteTexture(&dst_view, pixels, width * 4 * height, &layout,
                                 &size);

  wgpu::SamplerDescriptor sampler_desc = {};
  sampler_desc.minFilter = wgpu::FilterMode::Linear;
  sampler_desc.magFilter = wgpu::FilterMode::Linear;
  bd->render_resources.sampler = bd->device.CreateSampler(&sampler_desc);

  io.Fonts->SetTexID(
      static_cast<ImTextureID>(bd->render_resources.font_texture_view.Get()));
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

  CreateFontsTexture();
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

  ImGuiID font_tex_id_hash =
      ImHashData(&bd->render_resources.font_texture_view, sizeof(ImTextureID));
  bd->render_resources.image_bind_groups[font_tex_id_hash] =
      CreateImageBindGroup(bd->render_resources.image_bind_group_layout,
                           bd->render_resources.font_texture_view);
  return true;
}

void ImGui_ImplWGPU_InvalidateDeviceObjects() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd || !bd->device) return;

  bd->pipeline_state = nullptr;
  bd->render_resources = {};
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->SetTexID(nullptr);
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

  bd->init_info = *init_info;
  bd->device = init_info->device;
  bd->default_queue = init_info->device.GetQueue();
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

  ImGui_ImplWGPU_InvalidateDeviceObjects();
  bd->frame_resources.clear();
  io.BackendRendererName = nullptr;
  io.BackendRendererUserData = nullptr;
  io.BackendFlags &= ~ImGuiBackendFlags_RendererHasVtxOffset;
  delete bd;

  LOG(INFO) << "ImGui_ImplWGPU_Shutdown: Succeeded.";
}

void ImGui_ImplWGPU_NewFrame() {
  ImGui_ImplWGPU_Data* bd = ImGui_ImplWGPU_GetBackendData();
  if (!bd->pipeline_state) {
    ImGui_ImplWGPU_CreateDeviceObjects();
  }
}

#endif  // #ifndef IMGUI_DISABLE
