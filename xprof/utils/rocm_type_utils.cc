/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xprof/utils/rocm_type_utils.h"

#include <algorithm>
#include <optional>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"

namespace tensorflow {
namespace profiler {
namespace rocm {
namespace {

struct GpuFlopCapabilities {
  struct FlopCapabilityOnPrecisions {
    double fp64_tflops = 0;
    double fp32_tflops = 0;
    double tf32_tflops = 0;
    double bf16_tflops = 0;
    double fp16_tflops = 0;
    double fp8_tflops = 0;
    double mxfp6_tflops = 0;
    double mxfp4_tflops = 0;
    double int8_tops = 0;

    void ScaleWith(double scale) {
      fp64_tflops *= scale;
      fp32_tflops *= scale;
      tf32_tflops *= scale;
      bf16_tflops *= scale;
      fp16_tflops *= scale;
      fp8_tflops *= scale;
      mxfp6_tflops *= scale;
      mxfp4_tflops *= scale;
      int8_tops *= scale;
    }
  };

  FlopCapabilityOnPrecisions vector_unit;
  FlopCapabilityOnPrecisions matrix_unit;

  void ScaleWith(double scale) {
    vector_unit.ScaleWith(scale);
    matrix_unit.ScaleWith(scale);
  }
};

// ROCm XLA collector emits rocprofiler-sdk's agent name, composed via
// fmt::format("gfx{}{}{:x}", major, minor, step), e.g. "gfx942". 
// See rocprofiler-sdk source/lib/rocprofiler-sdk/agent.cpp.
// Reject a device name that does not follow this formatting.
absl::string_view GfxVersionIfWellFormed(absl::string_view device_name) {
  if (!absl::StartsWith(device_name, "gfx")) return "";
  absl::string_view suffix = device_name.substr(3);
  if (suffix.size() < 3) return "";
  for (char c : suffix) {
    if (!absl::ascii_isxdigit(c)) return "";
  }
  return device_name;
}

// Fallback for traces captured before the collector reported a device name.
// Resolves only when the major, minor pair can identify a unique architecture.
absl::string_view GfxVersionFromComputeCapability(int major, int minor) {
  if (major == 9 && minor == 4) return "gfx942";  // CDNA 3, MI300 series
  if (major == 9 && minor == 5) return "gfx950";  // CDNA 4, MI350 series
  return "";
}

absl::string_view ResolveAmdGfxVersion(const DeviceCapabilities& device_cap) {
  absl::string_view gfx = GfxVersionIfWellFormed(device_cap.device_name());
  if (!gfx.empty()) return gfx;
  return GfxVersionFromComputeCapability(
      device_cap.compute_capability().major(),
      device_cap.compute_capability().minor());
}

// LDS bytes per CU per clock.
// Each bank serves one read, write or atomic per cycle.
//
// AMD CDNA 4 architecture whitepaper, which additionally covers the LDS spec 
// for earlier generations (CDNA 3 and prior):
// https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
// 
// "The LDS in the AMD CDNA 3 architecture and prior 
// generations was a directly addressed structure with 32 banks, each containing 512 entries for 32-bits of 
// data – a total of 64KB of data. Each bank could read and write a 32-bit value and the LDS incorporates logic 
// for conflict detection and scheduling, a sophisticated crossbar and swizzle unit along with atomic execution units."
//
// RDNA series is absent as its LDS belongs to a workgroup processor rather than
// a CU. The published per-cycle figure cannot clearly be attributed per CU.
double GetAmdLdsBytesPerCuPerCycle(absl::string_view gfx_version) {
  static const auto& kTable = *new absl::btree_map<absl::string_view, double>{
      {"gfx908", 128.0},  // CDNA 1, 32 banks x 4 B
      {"gfx90a", 128.0},  // CDNA 2, 32 banks x 4 B
      {"gfx942", 128.0},  // CDNA 3, 32 banks x 4 B
      {"gfx950", 256.0},  // CDNA 4, 64 banks x 4 B
  };
  auto it = kTable.find(gfx_version);
  return it == kTable.end() ? 0.0 : it->second;
}

std::optional<GpuFlopCapabilities> GetAmdFlopCapsPerCuPerCycle(
    absl::string_view gfx_version) {
  static const auto& kTable =
      *new absl::btree_map<absl::string_view, GpuFlopCapabilities>{
          // MI100 (CDNA 1), Table 1 FLOPS/CLOCK/CU
          // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
          // https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi100.html
          {"gfx908",
           {.vector_unit = {.fp64_tflops = 64, .fp32_tflops = 128},
            .matrix_unit = {.fp32_tflops = 256,
                            .bf16_tflops = 512,
                            .fp16_tflops = 1024,
                            .int8_tops = 1024}}},
          // MI250X (CDNA 2), Table 1 FLOPS/CLOCK/CU.
          // https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
          // https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi250.html
          {"gfx90a",
           {.vector_unit = {.fp64_tflops = 128, .fp32_tflops = 128},
            .matrix_unit = {.fp64_tflops = 256,
                            .fp32_tflops = 256,
                            .bf16_tflops = 1024,
                            .fp16_tflops = 1024,
                            .int8_tops = 1024}}},
          // MI300X (CDNA 3), Table 1 FLOPS/CLOCK/CU
          // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
          // https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html
          {"gfx942",
           {.vector_unit = {.fp64_tflops = 128,
                            .fp32_tflops = 256,
                            .fp16_tflops = 256},
            .matrix_unit = {.fp64_tflops = 256,
                            .fp32_tflops = 256,
                            .tf32_tflops = 1024,
                            .bf16_tflops = 2048,
                            .fp16_tflops = 2048,
                            .fp8_tflops = 4096,
                            .int8_tops = 4096}}},
          // MI355X (CDNA 4), Table 1 FLOPS/CLOCK/CU
          // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
          {"gfx950",
           {.vector_unit = {.fp64_tflops = 128,
                            .fp32_tflops = 256,
                            .fp16_tflops = 256},
            .matrix_unit = {.fp64_tflops = 128,
                            .fp32_tflops = 256,
                            .bf16_tflops = 4096,
                            .fp16_tflops = 4096,
                            .fp8_tflops = 8192,
                            .mxfp6_tflops = 16384,
                            .mxfp4_tflops = 16384,
                            .int8_tops = 8192}}},
      };
  auto it = kTable.find(gfx_version);
  if (it == kTable.end()) return std::nullopt;
  return it->second;
}

}  // namespace

double GetFlopMaxThroughputPerCore(const DeviceCapabilities& device_cap) {
  absl::string_view gfx_version = ResolveAmdGfxVersion(device_cap);
  std::optional<GpuFlopCapabilities> cu_flops =
      GetAmdFlopCapsPerCuPerCycle(gfx_version);
  if (!cu_flops.has_value()) {
    LOG(WARNING) << "No FLOP rates known for AMD architecture '" << gfx_version
                 << "'; peak compute will be reported as zero.";
    return 0.0;
  }
  cu_flops->ScaleWith(device_cap.clock_rate_in_ghz());
  return std::max(
      {cu_flops->vector_unit.fp32_tflops, cu_flops->vector_unit.fp16_tflops,
       cu_flops->matrix_unit.fp32_tflops, cu_flops->matrix_unit.fp16_tflops});
}

double GetSharedMemoryBandwidthPerCore(const DeviceCapabilities& device_cap) {
  double bytes_per_cycle =
      GetAmdLdsBytesPerCuPerCycle(ResolveAmdGfxVersion(device_cap));
  if (bytes_per_cycle <= 0.0) return 0.0;
  return tsl::profiler::GigaToUni(bytes_per_cycle *
                                  device_cap.clock_rate_in_ghz());
}

std::string GpuModelName(const DeviceCapabilities& device_cap) {
  // Attempt to resolve exact gfx architecture. If no resolution
  // is reached, fall back to displaying the family (major) name.
  absl::string_view gfx = ResolveAmdGfxVersion(device_cap);
  if (!gfx.empty()) return absl::StrCat("AMD GPU - ", gfx);
  switch (device_cap.compute_capability().major()) {
    case 9:
      return "AMD GPU - gfx-9XX series";
    case 10:
      return "AMD GPU - gfx-10XX series";
    case 11:
      return "AMD GPU - gfx-11XX series";
    default:
      return "AMD GPU";
  }
}

}  // namespace rocm
}  // namespace profiler
}  // namespace tensorflow
