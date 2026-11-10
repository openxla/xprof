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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_GPU_LOW_OCCUPANCY_RULE_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_GPU_LOW_OCCUPANCY_RULE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/smart_suggestion/constants.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/smart_suggestion_rule.h"
#include "xprof/convert/source_info_utils.h"
#include "plugin/xprof/protobuf/kernel_stats.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {

// Rule to detect GPU kernel time spent in kernels with low theoretical
// occupancy (active warps per SM relative to the maximum, as computed from
// the launch configuration when the trace was captured).
//
// Only kernels with a known occupancy (> 0) count: traces from GPUs where the
// profiler cannot compute occupancy report 0 for every kernel, and the rule
// stays silent for them.
//
// Matrix multiplication and attention kernels do not count as low occupancy:
// tuned GEMM and attention kernels (e.g. warp-specialized WGMMA kernels on
// Hopper) often run at about 12.5% occupancy by design, and fewer registers or
// smaller tiles would make them slower.
class GpuLowOccupancyRule : public SmartSuggestionRule {
 public:
  // Kernel name substrings (matched without case) of matrix multiplication
  // and attention kernels from cuBLAS, cuBLASLt, CUTLASS, cuDNN and XLA
  // (Triton) fusions.
  static constexpr absl::string_view kMatmulOrAttentionNamePatterns[] = {
      "attention", "attn",  "cublas", "cutlass", "dot",
      "flash",     "fmha",  "gemm",   "gemv",    "matmul",
      "mha",       "nvjet", "sdpa",   "wgmma",   "xmma",
  };

  // Returns whether a report is a matrix multiplication or attention kernel,
  // by its TensorCore flag or by its name.
  static bool IsMatmulOrAttentionKernel(const KernelReport& report) {
    if (report.is_kernel_using_tensor_core()) return true;
    for (absl::string_view pattern : kMatmulOrAttentionNamePatterns) {
      if (absl::StrContainsIgnoreCase(report.name(), pattern)) return true;
    }
    return false;
  }

  // A kernel name with low occupancy. The kernel stats db has one report per
  // kernel name, launch configuration and op, so one kernel can have several.
  struct LowOccupancyKernel {
    // Low-occupancy time of all reports with this kernel name.
    uint64_t total_duration_ns = 0;
    // The longest low-occupancy report, whose launch parameters are listed.
    const KernelReport* longest_report = nullptr;
  };

  struct Analysis {
    // Kernel time with a known occupancy.
    uint64_t known_occupancy_ns = 0;
    // Kernel time with known occupancy below
    // kGpuLowOccupancyThresholdInPercent, without matrix multiplication and
    // attention kernels.
    uint64_t low_occupancy_ns = 0;
    // Low-occupancy time of matrix multiplication and attention kernels, which
    // is not in `low_occupancy_ns`.
    uint64_t matmul_low_occupancy_ns = 0;
    // Low-occupancy kernels, one per name, by total duration, longest first.
    std::vector<LowOccupancyKernel> low_occupancy_kernels;

    double LowOccupancyTimePercent() const {
      if (known_occupancy_ns == 0) return 0.0;
      return 100.0 * low_occupancy_ns / known_occupancy_ns;
    }

    double MatmulLowOccupancyTimePercent() const {
      if (known_occupancy_ns == 0) return 0.0;
      return 100.0 * matmul_low_occupancy_ns / known_occupancy_ns;
    }
  };

  static Analysis Analyze(const KernelStatsDb& kernel_stats_db) {
    Analysis analysis;
    absl::flat_hash_map<absl::string_view, size_t> index_by_name;
    for (const KernelReport& report : kernel_stats_db.reports()) {
      if (report.occupancy_pct() <= 0) continue;
      analysis.known_occupancy_ns += report.total_duration_ns();
      if (report.occupancy_pct() >= kGpuLowOccupancyThresholdInPercent) {
        continue;
      }
      if (IsMatmulOrAttentionKernel(report)) {
        analysis.matmul_low_occupancy_ns += report.total_duration_ns();
        continue;
      }
      analysis.low_occupancy_ns += report.total_duration_ns();
      auto [it, inserted] = index_by_name.try_emplace(
          report.name(), analysis.low_occupancy_kernels.size());
      if (inserted) analysis.low_occupancy_kernels.emplace_back();
      LowOccupancyKernel& kernel = analysis.low_occupancy_kernels[it->second];
      kernel.total_duration_ns += report.total_duration_ns();
      if (kernel.longest_report == nullptr ||
          report.total_duration_ns() >
              kernel.longest_report->total_duration_ns()) {
        kernel.longest_report = &report;
      }
    }
    std::stable_sort(
        analysis.low_occupancy_kernels.begin(),
        analysis.low_occupancy_kernels.end(),
        [](const LowOccupancyKernel& a, const LowOccupancyKernel& b) {
          return a.total_duration_ns > b.total_duration_ns;
        });
    return analysis;
  }

  bool MeetsConditions(const SignalProvider& signal_provider) const override {
    absl::StatusOr<const KernelStatsDb*> kernel_stats_db =
        signal_provider.GetKernelStatsDb();
    if (!kernel_stats_db.ok()) return false;
    return Analyze(**kernel_stats_db).LowOccupancyTimePercent() >=
           kGpuLowOccupancyTimeThresholdInPercent;
  }

  absl::StatusOr<std::optional<SmartSuggestion>> GenerateSuggestion(
      const SignalProvider& signal_provider) const override {
    TF_ASSIGN_OR_RETURN(const KernelStatsDb* kernel_stats_db,
                        signal_provider.GetKernelStatsDb());
    Analysis analysis = Analyze(*kernel_stats_db);

    std::string top_kernels;
    size_t num_listed = std::min<size_t>(analysis.low_occupancy_kernels.size(),
                                         kGpuLowOccupancyMaxKernelsListed);
    for (size_t i = 0; i < num_listed; ++i) {
      const LowOccupancyKernel& kernel = analysis.low_occupancy_kernels[i];
      const KernelReport& report = *kernel.longest_report;
      absl::StrAppend(
          &top_kernels, "<li><code>", EscapeHtml(report.name()),
          "</code>: ", absl::StrFormat("%.1f", report.occupancy_pct()),
          "% occupancy, ", report.registers_per_thread(), " registers/thread, ",
          report.static_shmem_bytes() + report.dynamic_shmem_bytes(),
          " bytes shared memory/block, ",
          absl::StrFormat("%.1f", 100.0 * kernel.total_duration_ns /
                                      analysis.known_occupancy_ns),
          "% of kernel time</li>");
    }

    std::string matmul_note;
    if (analysis.matmul_low_occupancy_ns > 0) {
      matmul_note = absl::StrCat(
          " Matrix multiplication and attention kernels with low occupancy (",
          absl::StrFormat("%.1f", analysis.MatmulLowOccupancyTimePercent()),
          "% of kernel time) are not counted, because tuned kernels of these "
          "types often run at low occupancy by design.");
    }

    SmartSuggestion suggestion;
    suggestion.set_rule_name("GpuLowOccupancyRule");
    suggestion.set_suggestion_text(absl::StrCat(
        "<p><b>", absl::StrFormat("%.1f", analysis.LowOccupancyTimePercent()),
        "%</b> of GPU kernel time is spent in kernels with a theoretical "
        "occupancy below <b>",
        absl::StrFormat("%.0f", kGpuLowOccupancyThresholdInPercent),
        "%</b>, other than matrix multiplication and attention kernels. Few "
        "resident warps per SM leave less room to hide memory and instruction "
        "latency.",
        matmul_note, " The longest such kernels are:</p><ul>", top_kernels,
        "</ul><p>Please consider the following optimizations:</p><ul>"
        "<li><b>Reduce registers per thread:</b> registers usually limit "
        "occupancy first. Use launch bounds or a register cap, or a smaller "
        "tile per thread.</li>"
        "<li><b>Reduce shared memory per block:</b> smaller tiles or fewer "
        "pipeline stages let more blocks reside on each SM.</li>"
        "<li><b>Revisit the launch configuration:</b> a grid with fewer blocks "
        "than SMs, or very small blocks, leaves SMs idle. For compiler-"
        "generated kernels, check whether autotuning picked this "
        "configuration.</li>"
        "</ul><p>Low occupancy is not always a problem: kernels with high "
        "instruction-level parallelism can reach peak throughput at low "
        "occupancy. Check achieved throughput for these kernels before "
        "changing them.</p>"));
    return suggestion;
  }
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_GPU_LOW_OCCUPANCY_RULE_H_
