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

#ifndef XPROF_CONVERT_FLAT_OP_METRICS_TO_RECORD_H_
#define XPROF_CONVERT_FLAT_OP_METRICS_TO_RECORD_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {

inline const auto& GetMetricsList(const FlatOpMetricsDb& db) {
  return db.op_instances();
}

inline absl::string_view GetMetricsName(const FlatOpMetrics& metrics) {
  return metrics.hlo_name();
}

inline double GigaFlopsPerSecondPerCore(const FlatOpMetrics& metrics) {
  return tsl::profiler::SafeDivide(
      metrics.flops_v2(), tsl::profiler::PicoToNano(metrics.time_ps()));
}

inline double GigaFlopsPerSecondPerCoreNormalizedOnDvfs(
    const FlatOpMetrics& metrics) {
  if (metrics.normalized_time_ps() == 0) {
    return GigaFlopsPerSecondPerCore(metrics);
  }
  return GigaFlopsPerSecondPerCore(metrics) *
         (tsl::profiler::SafeDivide(metrics.normalized_time_ps(),
                                    metrics.time_ps()));
}

inline double GigaModelFlopsPerSecondPerCore(const FlatOpMetrics& metrics) {
  return tsl::profiler::SafeDivide(
      metrics.model_flops_v2(), tsl::profiler::PicoToNano(metrics.time_ps()));
}

inline double BytesAccessedPerCore(
    const FlatOpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType operation_type) {
  uint64_t bytes = 0;
  if (memory_space == MemorySpace::MEMORY_SPACE_ALL) {
    bytes = metrics.bytes_accessed();
  } else {
    for (const auto& breakdown : metrics.memory_accessed_breakdown()) {
      if ((breakdown.operation_type() != operation_type) &&
          (operation_type != OpMetrics::MemoryAccessed::UNKNOWN)) {
        continue;
      }
      if (((memory_space == MemorySpace::MEMORY_SPACE_HBM) &&
           (breakdown.memory_space() == MemorySpace::MEMORY_SPACE_HBM)) ||
          ((memory_space == MemorySpace::MEMORY_SPACE_ON_CHIP) &&
           (breakdown.memory_space() != MemorySpace::MEMORY_SPACE_HBM))) {
        bytes += breakdown.bytes_accessed();
      }
    }
  }
  return bytes;
}

inline double GigaBytesPerSecondPerCore(
    const FlatOpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType operation_type) {
  return tsl::profiler::SafeDivide(
      BytesAccessedPerCore(metrics, memory_space, operation_type),
      tsl::profiler::PicoToNano(metrics.time_ps()));
}

inline double GibiBytesPerSecondPerCore(
    const FlatOpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType op_type) {
  return tsl::profiler::GigaToGibi(
      GigaBytesPerSecondPerCore(metrics, memory_space, op_type));
}

template <typename Record>
inline void SetExecutionTimes(const FlatOpMetrics& metrics, Record* record) {
  record->set_occurrences(metrics.occurrences());
  record->set_total_time_in_us(tsl::profiler::PicoToMicro(metrics.time_ps()));
  record->set_avg_time_in_us(tsl::profiler::SafeDivide(
      record->total_time_in_us(), metrics.occurrences()));
  record->set_total_self_time_in_us(
      tsl::profiler::PicoToMicro(metrics.self_time_ps()));
  record->set_avg_self_time_in_us(tsl::profiler::SafeDivide(
      record->total_self_time_in_us(), metrics.occurrences()));
}

template <typename Record>
inline void SetTpuUnitFractions(const FlatOpMetrics& metrics, Record* record) {
  record->set_dma_stall_fraction(
      tsl::profiler::SafeDivide(metrics.dma_stall_ps(), metrics.time_ps()));
}

// Returns a sorted vector of pointers to FlatOpMetrics in the given database.
// The returned pointers are only valid as long as `metrics_db` exists and is
// not modified.
std::vector<const FlatOpMetrics*> SortedOpMetricsDb(
    const FlatOpMetricsDb& metrics_db, int max_records = -1);

template <typename Record>
inline void SetRooflineMetrics(const FlatOpMetrics& metrics,
                               const PerfEnv& perf_env,
                               const RunEnvironment& run_env, Record* record,
                               bool apply_time_scale_factor = false) {
  using ::tensorflow::profiler::MemorySpace;
  using ::tensorflow::profiler::PerformanceInfo;

  record->set_measured_flop_rate(GigaFlopsPerSecondPerCore(metrics));
  record->set_model_flop_rate(GigaModelFlopsPerSecondPerCore(metrics));
  record->set_measured_memory_bw(GibiBytesPerSecondPerCore(
      metrics, tensorflow::profiler::MemorySpace::MEMORY_SPACE_ALL,
      OpMetrics::MemoryAccessed::UNKNOWN));
  record->set_flops(metrics.flops());
  record->set_flops_v2(metrics.flops_v2());
  record->set_bytes_accessed(metrics.bytes_accessed());
  record->set_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops_v2(), metrics.bytes_accessed()));

  uint64_t hbm_bytes = 0;
  uint64_t cmem_read_bytes = 0;
  uint64_t cmem_write_bytes = 0;
  uint64_t vmem_read_bytes = 0;
  uint64_t vmem_write_bytes = 0;
  for (const auto& memory_access : metrics.memory_accessed_breakdown()) {
    if (memory_access.memory_space() == PerformanceInfo::MemoryAccessed::HBM) {
      hbm_bytes += memory_access.bytes_accessed();
    } else if (memory_access.memory_space() ==
               PerformanceInfo::MemoryAccessed::CMEM) {
      if (memory_access.operation_type() == OpMetrics::MemoryAccessed::READ) {
        cmem_read_bytes += memory_access.bytes_accessed();
      } else if (memory_access.operation_type() ==
                 OpMetrics::MemoryAccessed::WRITE) {
        cmem_write_bytes += memory_access.bytes_accessed();
      }
    } else if (memory_access.memory_space() ==
               PerformanceInfo::MemoryAccessed::VMEM) {
      if (memory_access.operation_type() == OpMetrics::MemoryAccessed::READ) {
        vmem_read_bytes += memory_access.bytes_accessed();
      } else if (memory_access.operation_type() ==
                 OpMetrics::MemoryAccessed::WRITE) {
        vmem_write_bytes += memory_access.bytes_accessed();
      }
    }
  }
  if (metrics.memory_accessed_breakdown_size() == 0) {
    hbm_bytes = metrics.bytes_accessed();
  }
  int64_t device_time_ps = apply_time_scale_factor
                               ? metrics.normalized_time_ps()
                               : metrics.time_ps();
  record->set_hbm_bw(tsl::profiler::GibibytesPerSecond(
      hbm_bytes, tsl::profiler::PicoToNano(metrics.time_ps())));
  record->set_cmem_read_bw(tsl::profiler::GibibytesPerSecond(
      cmem_read_bytes, tsl::profiler::PicoToNano(device_time_ps)));
  record->set_cmem_write_bw(tsl::profiler::GibibytesPerSecond(
      cmem_write_bytes, tsl::profiler::PicoToNano(device_time_ps)));
  record->set_vmem_read_bw(tsl::profiler::GibibytesPerSecond(
      vmem_read_bytes, tsl::profiler::PicoToNano(device_time_ps)));
  record->set_vmem_write_bw(tsl::profiler::GibibytesPerSecond(
      vmem_write_bytes, tsl::profiler::PicoToNano(device_time_ps)));
  record->set_hbm_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops_v2(), hbm_bytes));
  record->set_cmem_read_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops_v2(), cmem_read_bytes));
  record->set_cmem_write_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops_v2(), cmem_write_bytes));
  record->set_vmem_read_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops_v2(), vmem_read_bytes));
  record->set_vmem_write_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops_v2(), vmem_write_bytes));

  constexpr absl::string_view kUnknown = "Unknown";
  constexpr absl::string_view kCompute = "Compute";
  constexpr absl::string_view kHbm = "HBM";
  constexpr absl::string_view kCmemRead = "CMEM Read";
  constexpr absl::string_view kCmemWrite = "CMEM Write";
  constexpr absl::string_view kVmemRead = "VMEM Read";
  constexpr absl::string_view kVmemWrite = "VMEM Write";
  constexpr absl::string_view kShmL1 = "Shm/L1";

  absl::string_view bottleneck_resource = kUnknown;
  double bottleneck_utilization = 0;
  double bottleneck_operational_intensity = 0;
  double peak_flops =
      tsl::profiler::TeraToGiga(perf_env.peak_tera_flops_per_second());
  double flops_utilization =
      tsl::profiler::SafeDivide(record->measured_flop_rate(), peak_flops);
  if (bottleneck_utilization < flops_utilization) {
    bottleneck_resource = kCompute;
    bottleneck_utilization = flops_utilization;
    bottleneck_operational_intensity = record->operational_intensity();
  }
  double peak_hbm_bw = GetMemoryPeakBandwidth(perf_env, 0);
  double hbm_bw_utilization = tsl::profiler::SafeDivide(
      record->hbm_bw(), tsl::profiler::GigaToGibi(peak_hbm_bw));
  if (bottleneck_utilization < hbm_bw_utilization) {
    bottleneck_resource = kHbm;
    bottleneck_utilization = hbm_bw_utilization;
    bottleneck_operational_intensity = record->hbm_operational_intensity();
  }
  tensorflow::profiler::HardwareType hardware_type = run_env.hardware_type();
  if (hardware_type == tensorflow::profiler::HardwareType::TPU) {
    if (cmem_read_bytes) {
      double peak_cmem_read_bw = GetMemoryPeakBandwidth(perf_env, 3);
      if (peak_cmem_read_bw) {
        double cmem_read_bw_utilization = tsl::profiler::SafeDivide(
            record->cmem_read_bw(),
            tsl::profiler::GigaToGibi(peak_cmem_read_bw));
        if (bottleneck_utilization < cmem_read_bw_utilization) {
          bottleneck_resource = kCmemRead;
          bottleneck_utilization = cmem_read_bw_utilization;
          bottleneck_operational_intensity =
              record->cmem_read_operational_intensity();
        }
      }
    }
    if (cmem_write_bytes) {
      double peak_cmem_write_bw = GetMemoryPeakBandwidth(perf_env, 4);
      if (peak_cmem_write_bw) {
        double cmem_write_bw_utilization = tsl::profiler::SafeDivide(
            record->cmem_write_bw(),
            tsl::profiler::GigaToGibi(peak_cmem_write_bw));
        if (bottleneck_utilization < cmem_write_bw_utilization) {
          bottleneck_resource = kCmemWrite;
          bottleneck_utilization = cmem_write_bw_utilization;
          bottleneck_operational_intensity =
              record->cmem_write_operational_intensity();
        }
      }
    }
    if (vmem_read_bytes) {
      double peak_vmem_read_bw = GetMemoryPeakBandwidth(perf_env, 5);
      if (peak_vmem_read_bw) {
        double vmem_read_bw_utilization = tsl::profiler::SafeDivide(
            record->vmem_read_bw(),
            tsl::profiler::GigaToGibi(peak_vmem_read_bw));
        if (bottleneck_utilization < vmem_read_bw_utilization) {
          bottleneck_resource = kVmemRead;
          bottleneck_utilization = vmem_read_bw_utilization;
          bottleneck_operational_intensity =
              record->vmem_read_operational_intensity();
        }
      }
    }
    if (vmem_write_bytes) {
      double peak_vmem_write_bw = GetMemoryPeakBandwidth(perf_env, 6);
      if (peak_vmem_write_bw) {
        double vmem_write_bw_utilization = tsl::profiler::SafeDivide(
            record->vmem_write_bw(),
            tsl::profiler::GigaToGibi(peak_vmem_write_bw));
        if (bottleneck_utilization < vmem_write_bw_utilization) {
          bottleneck_resource = kVmemWrite;
          bottleneck_utilization = vmem_write_bw_utilization;
          bottleneck_operational_intensity =
              record->vmem_write_operational_intensity();
        }
      }
    }
  }
  if (hardware_type == tensorflow::profiler::HardwareType::GPU) {
    double peak_shm_l1_bw = GetMemoryPeakBandwidth(perf_env, 2);
    if (peak_shm_l1_bw) {
      double shm_l1_bw_utilization = tsl::profiler::SafeDivide(
          record->hbm_bw(), tsl::profiler::GigaToGibi(peak_shm_l1_bw));
      if (bottleneck_utilization < shm_l1_bw_utilization) {
        bottleneck_resource = kShmL1;
        bottleneck_utilization = shm_l1_bw_utilization;
        bottleneck_operational_intensity = record->hbm_operational_intensity();
      }
    }
  }
  record->set_bound_by(std::string(bottleneck_resource));
  record->set_bottleneck_operational_intensity(
      bottleneck_operational_intensity);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_FLAT_OP_METRICS_TO_RECORD_H_
