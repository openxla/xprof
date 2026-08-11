/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/utils/hardware_type_utils.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

// The calculation methods is referred from Nvidia developer forum:
// https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727
// Below data are calculated from the various NVidia whitepapers/specs.

// https://resources.nvidia.com/en-us-blackwell-architecture?ncid=pa-srch-goog-585983-Intel-Brand-Broad
// Dense Compute as default.
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_10_0 = {
    .vector_unit =
        {
            .fp64_tflops = 148,
            .fp32_tflops = 296,
            .bf16_tflops = 592,
            .fp16_tflops = 592,
            .int8_tops = 1184,
        },
    .matrix_unit =
        {
            .fp64_tflops = 148,
            .fp32_tflops = 4096,
            .bf16_tflops = 8192,
            .fp16_tflops = 8192,
            .fp8_tflops = 16384,
            .int8_tops = 16384,
        },
    .has_matrix_unit_sparsity_support = true,
};

// https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_9_0 = {
    .vector_unit =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 256,
            .bf16_tflops = 512,
            .fp16_tflops = 512,
            .int8_tops = 1024,
        },
    .matrix_unit =
        {
            .fp64_tflops = 256,
            .fp32_tflops = 2048,
            .bf16_tflops = 4096,
            .fp16_tflops = 4096,
            .fp8_tflops = 8192,
            .int8_tops = 8192,
        },
    .has_matrix_unit_sparsity_support = true,
};

// https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_8_9 = {
    .vector_unit =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 256,
            .bf16_tflops = 256,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .matrix_unit =
        {
            .fp32_tflops = 512,
            .bf16_tflops = 1024,
            .fp16_tflops = 1024,
            .fp8_tflops = 2048,
            .int8_tops = 2048,
            .int4_tops = 4096,
        },
    .has_matrix_unit_sparsity_support = true,
};

// https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_8_6 = {
    .vector_unit =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 256,
            .bf16_tflops = 256,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .matrix_unit =
        {
            .fp32_tflops = 256,
            .bf16_tflops = 512,
            .fp16_tflops = 1024,
            .int8_tops = 2048,
            .int4_tops = 4096,
        },
    .has_matrix_unit_sparsity_support = true,
};

// https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_8_0 = {
    .vector_unit =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .bf16_tflops = 256,
            .fp16_tflops = 512,
            .int8_tops = 512,
        },
    .matrix_unit =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 1024,
            .bf16_tflops = 2048,
            .fp16_tflops = 2048,
            .int8_tops = 4096,
        },
    .has_matrix_unit_sparsity_support = true,
};

// https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_7_5 = {
    .vector_unit =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .matrix_unit =
        {
            .fp16_tflops = 1024,
            .int8_tops = 2048,
            .int4_tops = 4096,
        },
    .has_matrix_unit_sparsity_support = false,
};

// https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_7_0 = {
    .vector_unit =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .bf16_tflops = 0.0,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .matrix_unit =
        {
            .fp16_tflops = 1024,
        },
    .has_matrix_unit_sparsity_support = false,
};

// https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_6_1 = {
    .vector_unit =
        {
            .fp64_tflops = 8,
            .fp32_tflops = 256,
            .fp16_tflops = 4,
            .int8_tops = 1024,
        },
    .matrix_unit = {},
    .has_matrix_unit_sparsity_support = false,
};

// https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_6_0 = {
    .vector_unit =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .matrix_unit = {},
    .has_matrix_unit_sparsity_support = false,
};

// https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_5_0 = {
    .vector_unit =
        {
            .fp64_tflops = 4,
            .fp32_tflops = 256,
        },
    .matrix_unit = {},
    .has_matrix_unit_sparsity_support = false,
};

// https://www.nvidia.com/content/PDF/product-specifications/GeForce_GTX_680_Whitepaper_FINAL.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_3_0 = {
    .vector_unit =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 384,
        },
    .matrix_unit = {},
    .has_matrix_unit_sparsity_support = false,
};

const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_2_0 = {
    .vector_unit =
        {
            .fp64_tflops = 8,
            .fp32_tflops = 64,
        },
    .matrix_unit = {},
    .has_matrix_unit_sparsity_support = false,
};

// Extracts the AMD architecture from a reported device name.
//
// ROCm collector emits rocprofiler-sdk's agent name, composed via
// fmt::format("gfx{}{}{:x}", major, minor, step), so it is a bare processor name
// such as "gfx942". See rocprofiler-sdk source/lib/rocprofiler-sdk/agent.cpp:
// https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/rocprofiler-sdk/agent.cpp#L769-L775
//
// HIP's hipDeviceProp_t::gcnArchName instead carries the target-ID form,
// "gfx942:sramecc+:xnack-", so keep only the processor.
absl::string_view GfxVersionFromDeviceName(absl::string_view device_name) {
  return device_name.substr(0, device_name.find(':'));
}

// Fallback for traces captured before the collector reported a device name.
// Only pairs identifying exactly one architecture are listed: ROCm derives the
// capability from gfx_target_version and drops the step digit, so (9, 0) covers
// both gfx908 (MI100) and gfx90a (MI200/250), whose rates differ.
absl::string_view GfxVersionFromComputeCapability(int major, int minor) {
  if (major == 9 && minor == 4) return "gfx942";  // CDNA 3, MI300 series
  if (major == 9 && minor == 5) return "gfx950";  // CDNA 4, MI350 series
  return "";
}

// Resolves the architecture used to key the AMD tables below: the reported
// device name if there is one, else the compute capability.
absl::string_view ResolveAmdGfxVersion(const DeviceCapabilities& device_cap) {
  absl::string_view gfx = GfxVersionFromDeviceName(device_cap.device_name());
  if (!gfx.empty()) return gfx;
  return GfxVersionFromComputeCapability(
      device_cap.compute_capability().major(),
      device_cap.compute_capability().minor());
}

// LDS bytes per CU per clock. Each bank serves one read, write or atomic per
// cycle, so the figure is the same in both directions.
//
// AMD CDNA 4 architecture whitepaper, which additionally covers LDS spec for
// earlier generations (CDNA 3 and prior):
// https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
//
// Note RDNA series is absent as its LDS belongs to a workgroup processor rather
// than a CU. The published per-cycle figure cannot clearly be attributed per CU.
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

GpuFlopCapabilities GetNvidiaFlopCapsPerSMPerCycle(int major_comp_cap,
                                                   int minor_comp_cap) {
  static const auto& kPerSMFlopCapsTable =
      *new absl::btree_map<int, GpuFlopCapabilities const*>{
          // TODO: Add newer GPUS when available.
          {10000, &kComputeCap_PerSM_PerCycle_10_0},
          {9000, &kComputeCap_PerSM_PerCycle_9_0},
          {8090, &kComputeCap_PerSM_PerCycle_8_9},
          {8060, &kComputeCap_PerSM_PerCycle_8_6},
          {8000, &kComputeCap_PerSM_PerCycle_8_0},
          {7050, &kComputeCap_PerSM_PerCycle_7_5},
          {7000, &kComputeCap_PerSM_PerCycle_7_0},
          {6010, &kComputeCap_PerSM_PerCycle_6_1},
          {6000, &kComputeCap_PerSM_PerCycle_6_0},
          {5000, &kComputeCap_PerSM_PerCycle_5_0},
          {3000, &kComputeCap_PerSM_PerCycle_3_0},
          {2000, &kComputeCap_PerSM_PerCycle_2_0},
      };

  // TODO(b/409612464): - Need more discussion on how to handle the case when
  // the compute cap is not found in above table. Currently we back off to the
  // highest compute cap less than the given compute cap, or the oldest compute
  // cap in the table if the given compute cap is even older than it. Another
  // way is to just report not found and return 0 in GpuFlopCapabilities. Also
  // more fine-grained back off also could be used.
  const int normalized_compute_cap =
      major_comp_cap * 1000 + minor_comp_cap * 10;
  auto it = kPerSMFlopCapsTable.lower_bound(normalized_compute_cap);
  if (it == kPerSMFlopCapsTable.end() || it->first > normalized_compute_cap) {
    if (it != kPerSMFlopCapsTable.begin()) it = std::prev(it);
    LOG(WARNING) << "GPU compute capability " << major_comp_cap << "."
                 << minor_comp_cap
                 << " is not found. Use the highest compute cap known "
                 << (it->first / 1000) << "." << ((it->first % 1000) / 10)
                 << " instead.";
  }
  return GpuFlopCapabilities(*(it->second));
}

// Peak FLOPs per CU per cycle. FMA considered as 2 FLOPs. Matrix rates are dense
// MFMA and vector rates are VALU. CDNA 3 and 4 are stated directly by the CDNA 4
// whitepaper (Table 1, in these units); CDNA 1 and 2 are derived from the
// published peak as: peak_flops / (cu_count * clock). Pinned by tests either way.
//
// fp16 is set alongside bf16 as GetFlopMaxThroughputPerSM considers only fp32
// and fp16. A matrix rate left solely in bf16_tflops is skipped and the peak
// falls back to vector fp32.
std::optional<GpuFlopCapabilities> GetAmdFlopCapsPerCuPerCycle(
    absl::string_view gfx_version) {
  static const auto& kTable =
      *new absl::btree_map<absl::string_view, GpuFlopCapabilities>{
          // MI100 (CDNA 1), 120 CU at 1.502 GHz: FP64 11.5, FP32 23.1,
          // BF16 92.3, FP16 184.6 TFLOPS. bf16 MFMA is half rate on CDNA 1.
          // https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi100.html
          {"gfx908",
           {.vector_unit = {.fp64_tflops = 64, .fp32_tflops = 128},
            .matrix_unit = {.bf16_tflops = 512, .fp16_tflops = 1024}}},
          // MI250X (CDNA 2), 104 CU per GCD at 1.7 GHz: FP64 45.3, FP32 45.3,
          // FP16 = BF16 362.1 TFLOPS per GCD. Full-rate FP64.
          // https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi250.html
          {"gfx90a",
           {.vector_unit = {.fp64_tflops = 256, .fp32_tflops = 256},
            .matrix_unit = {.bf16_tflops = 2048, .fp16_tflops = 2048}}},
          // MI300X (CDNA 3). CDNA 4 whitepaper Table 1, MI300X column.
          // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
          {"gfx942",
           {.vector_unit = {.fp64_tflops = 128, .fp32_tflops = 256},
            .matrix_unit = {.bf16_tflops = 2048,
                            .fp16_tflops = 2048,
                            .fp8_tflops = 4096}}},
          // MI355X (CDNA 4). CDNA 4 whitepaper Table 1, MI355X column.
          {"gfx950",
           {.vector_unit = {.fp64_tflops = 128, .fp32_tflops = 256},
            .matrix_unit = {.bf16_tflops = 4096,
                            .fp16_tflops = 4096,
                            .fp8_tflops = 8192}}},
      };
  auto it = kTable.find(gfx_version);
  if (it == kTable.end()) return std::nullopt;
  return it->second;
}

GpuFlopCapabilities GetGpuFlopCapabilitiesPerSM(
    const DeviceCapabilities& device_cap) {
  GpuFlopCapabilities flops_cap{};
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
    flops_cap =
        GetNvidiaFlopCapsPerSMPerCycle(device_cap.compute_capability().major(),
                                       device_cap.compute_capability().minor());
  } else if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorAMD) {
    absl::string_view gfx_version = ResolveAmdGfxVersion(device_cap);
    if (std::optional<GpuFlopCapabilities> amd_flops =
            GetAmdFlopCapsPerCuPerCycle(gfx_version)) {
      flops_cap = *amd_flops;
    } else {
      LOG(WARNING) << "No FLOP rates known for AMD architecture '" << gfx_version
                   << "'; peak compute will be reported as zero.";
    }
  } else {
    LOG(WARNING) << "Unsupported device vendor " << device_cap.device_vendor();
  }

  flops_cap.ScaleWith(device_cap.clock_rate_in_ghz());
  return flops_cap;
}

}  // namespace

double GetFlopMaxThroughputPerSM(const DeviceCapabilities& device_cap) {
  GpuFlopCapabilities sm_flops = GetGpuFlopCapabilitiesPerSM(device_cap);
  double result = std::max(
      {sm_flops.vector_unit.fp32_tflops, sm_flops.vector_unit.fp16_tflops,
       sm_flops.matrix_unit.fp32_tflops, sm_flops.matrix_unit.fp16_tflops});
  VLOG(3) << "GetFlopMaxThroughputPerSM get result: " << result << " GFLOPs";
  return result;
}

double GetSharedMemoryBandwidthPerSM(const DeviceCapabilities& device_cap) {
  double transaction_byts_per_cycle = 0.0;
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
    // https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsshared.htm
    // Compute capability 2.0, each bank has bandwidth of 4 bytes per 2 cycles.
    // For compute capability 3.0 and above, each bank has bandwidth 8 bytes per
    // cycle. Each SM has 32 banks.
    transaction_byts_per_cycle =
        device_cap.compute_capability().major() <= 2 ? (32 * 4 / 2) : (32 * 8);
  } else if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorAMD) {
    transaction_byts_per_cycle =
        GetAmdLdsBytesPerCuPerCycle(ResolveAmdGfxVersion(device_cap));
  }
  // An unknown vendor or architecture reports nothing rather than a value
  // borrowed from hardware it does not describe.
  if (transaction_byts_per_cycle <= 0.0) return 0.0;
  double GiBPS = transaction_byts_per_cycle * device_cap.clock_rate_in_ghz();
  return tsl::profiler::GigaToUni(GiBPS);
}

std::string GpuModelName(const DeviceCapabilities& device_cap) {
  if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorNvidia) {
    switch (device_cap.compute_capability().major()) {
      case 2:
        return "Nvidia GPU (Fermi)";
      case 3:
        return "Nvidia GPU (Kepler)";
      case 5:
        return "Nvidia GPU (Maxwell)";
      case 6:
        return "Nvidia GPU (Pascal)";
      case 7:
        if (device_cap.compute_capability().minor() < 5) {
          return "Nvidia GPU (Volta)";
        } else {
          return "Nvidia GPU (Turing)";
        }
      case 8:
        if (device_cap.compute_capability().minor() < 9) {
          return "Nvidia GPU (Ampere)";
        } else {
          return "Nvidia GPU (Ada Lovelace)";
        }
      case 9:
        return "Nvidia GPU (Hopper)";
      case 10:
        return "Nvidia GPU (Blackwell)";
      default:
        return "Nvidia GPU";
    }
  } else if (device_cap.device_vendor() == tsl::profiler::kDeviceVendorAMD) {
    // Prefer ROCm provided GCN architecture name if provided.
    // Otherwise, fallback to existing name derivation via major/minor.
    absl::string_view gfx = GfxVersionFromDeviceName(device_cap.device_name());
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
  } else {
    LOG(ERROR) << "Unknown device vendor " << device_cap.device_vendor();
    return "";
  }
}

HardwareType ParseHardwareType(absl::string_view device_type) {
  if (absl::StrContains(device_type, "GPU")) return HardwareType::GPU;
  if (device_type == "CPU") return HardwareType::CPU_ONLY;
  if (absl::StrContains(device_type, "TPU")) return HardwareType::TPU;
  return HardwareType::UNKNOWN_HARDWARE;
}

bool HasDevice(HardwareType x) { return x > tensorflow::profiler::CPU_ONLY; }

}  // namespace profiler
}  // namespace tensorflow
