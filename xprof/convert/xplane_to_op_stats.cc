/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_op_stats.h"

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/duty_cycle_combiner.h"
#include "xprof/convert/duty_cycle_tracker.h"
#include "xprof/convert/model_tracker.h"
#include "xprof/convert/op_metrics_db_combiner.h"
#include "xprof/convert/step_events_to_steps_db.h"
#include "xprof/convert/xplane_to_kernel_stats_db.h"
#include "xprof/convert/xplane_to_op_metrics_db.h"
#include "xprof/convert/xplane_to_step_events.h"
#include "xprof/convert/xplane_to_tf_functions.h"
#include "xprof/convert/xprof_thread_pool_executor.h"
#include "plugin/xprof/protobuf/diagnostics.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "plugin/xprof/protobuf/tf_function.pb.h"
#include "xprof/utils/device_caps_utils.h"
#include "xprof/utils/event_span.h"
#include "xprof/utils/gpu_event_stats.h"
#include "xprof/utils/hardware_type_utils.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/hlo_proto_map.h"
#include "xprof/utils/kernel_stats_utils.h"
#include "xprof/utils/op_utils.h"
#include "xprof/utils/xprof_gpu_cost_analysis_types.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::FindTensorCorePlanes;
using ::tsl::profiler::kGpuPlanePrefix;
using ::tsl::profiler::kTpuPlanePrefix;
using tsl::profiler::Timespan;
using ::tsl::profiler::XPlaneBuilder;

std::string Hostname(const XSpace& space) {
  if (space.hostnames().empty()) return "localhost";
  DCHECK_EQ(space.hostnames_size(), 1);
  const std::string& hostname = space.hostnames(0);
  return hostname;
}

}  // namespace

PerfEnv MakePerfEnv(double peak_tera_flops_per_second,
                    std::vector<double> peak_bws) {
  PerfEnv result;
  result.set_peak_tera_flops_per_second(peak_tera_flops_per_second);

  for (const auto bw : peak_bws) {
    result.add_peak_bws_giga_bytes_per_second(bw);
  }
  result.set_ridge_point(tsl::profiler::TeraToGiga(peak_tera_flops_per_second) /
                         peak_bws[MemBwType::MEM_BW_TYPE_HBM_RW]);
  return result;
}

PerfEnv MakePerfEnvForTpu(double peak_tera_flops_per_second,
                          std::vector<double> peak_bws, bool has_merged_vmem,
                          bool has_megacore) {
  PerfEnv result = MakePerfEnv(peak_tera_flops_per_second, peak_bws);
  result.set_has_cmem(peak_bws[MemBwType::MEM_BW_TYPE_CMEM_RD] > 0 ||
                      peak_bws[MemBwType::MEM_BW_TYPE_CMEM_WR] > 0);
  result.set_has_merged_vmem(has_merged_vmem);
  result.set_has_megacore(has_megacore);
  return result;
}

PerfEnv MakePerfEnvForGpu(double peak_tera_flops_per_second,
                          std::vector<double> peak_bws) {
  return MakePerfEnv(peak_tera_flops_per_second, peak_bws);
}

PerfEnv GetPerfEnvFromXPlane(const XPlane& device_plane) {
  DeviceCapabilities cap = GetDeviceCaps(device_plane);
  if (!absl::StartsWith(device_plane.name(), kTpuPlanePrefix)) {
    double peak_tera_flops_per_second =
        cap.num_cores() *
        tsl::profiler::GigaToTera(GetFlopMaxThroughputPerSM(cap));
    double hbm_bw_giga_bytes_per_second =
        tsl::profiler::UniToGiga(cap.memory_bandwidth());
    double shm_giga_bytes_per_second =
        cap.num_cores() *
        tsl::profiler::UniToGiga(GetSharedMemoryBandwidthPerSM(cap));
    // Note that treat SRAM_RD and SRAM_WR as the same. So in future, we could
    // only use one for shared memory / L1 cache, one for another like L2.
    return MakePerfEnvForGpu(peak_tera_flops_per_second,
                             {/*HBM_RW=*/hbm_bw_giga_bytes_per_second,
                              /*SRAM_RD=*/shm_giga_bytes_per_second,
                              /*SRAM_WR=*/shm_giga_bytes_per_second});
  } else {
    XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(&device_plane);
    std::optional<XStatVisitor> peak_tera_flops_per_second =
        visitor.GetStat(StatType::kDevCapPeakTeraflopsPerSecond);
    double peak_tera_flops_per_second_val =
        peak_tera_flops_per_second.has_value()
            ? peak_tera_flops_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> peak_hbm_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakHbmBwGigabytesPerSecond);
    double peak_hbm_bw_giga_bytes_per_second_val =
        peak_hbm_bw_giga_bytes_per_second.has_value()
            ? peak_hbm_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> peak_sram_rd_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakSramRdBwGigabytesPerSecond);
    double peak_sram_rd_bw_giga_bytes_per_second_val =
        peak_sram_rd_bw_giga_bytes_per_second.has_value()
            ? peak_sram_rd_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> peak_sram_wr_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakSramWrBwGigabytesPerSecond);
    double peak_sram_wr_bw_giga_bytes_per_second_val =
        peak_sram_wr_bw_giga_bytes_per_second.has_value()
            ? peak_sram_wr_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> cmem_rd_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakCmemRdBwGigabytesPerSecond);
    double cmem_rd_bw_giga_bytes_per_second_val =
        cmem_rd_bw_giga_bytes_per_second.has_value()
            ? cmem_rd_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> cmem_wr_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakCmemWrBwGigabytesPerSecond);
    double cmem_wr_bw_giga_bytes_per_second_val =
        cmem_wr_bw_giga_bytes_per_second.has_value()
            ? cmem_wr_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> vmem_rd_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakVmemRdBwGigabytesPerSecond);
    double vmem_rd_bw_giga_bytes_per_second_val =
        vmem_rd_bw_giga_bytes_per_second.has_value()
            ? vmem_rd_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> vmem_wr_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakVmemWrBwGigabytesPerSecond);
    double vmem_wr_bw_giga_bytes_per_second_val =
        vmem_wr_bw_giga_bytes_per_second.has_value()
            ? vmem_wr_bw_giga_bytes_per_second->DoubleValue()
            : 0.0;
    std::optional<XStatVisitor> has_megacore =
        visitor.GetStat(StatType::kDevHasMegacore);
    bool has_megacore_val =
        has_megacore.has_value() ? has_megacore->BoolValue() : false;
    std::optional<XStatVisitor> has_merged_vmem =
        visitor.GetStat(StatType::kDevHasMergedVmem);
    bool has_merged_vmem_val =
        has_merged_vmem.has_value() ? has_merged_vmem->BoolValue() : false;
    return MakePerfEnvForTpu(
        peak_tera_flops_per_second_val,
        {/*HBM_RW=*/peak_hbm_bw_giga_bytes_per_second_val,
         /*SRAM_RD=*/peak_sram_rd_bw_giga_bytes_per_second_val,
         /*SRAM_WR=*/peak_sram_wr_bw_giga_bytes_per_second_val,
         /**CMEM_RD=*/cmem_rd_bw_giga_bytes_per_second_val,
         /**CMEM_WR=*/cmem_wr_bw_giga_bytes_per_second_val,
         /**VMEM_RD=*/vmem_rd_bw_giga_bytes_per_second_val,
         /**VMEM_WR=*/vmem_wr_bw_giga_bytes_per_second_val},
        has_merged_vmem_val, has_megacore_val);
  }
}

void SetRunEnvironment(const XSpace& space, RunEnvironment* env) {
  // Currently, we only support profiling one host and one program.
  env->set_host_count(1);
  env->set_task_count(1);
  env->mutable_hostnames()->insert({Hostname(space), true});

  std::vector<const XPlane*> gpu_planes =
      FindPlanesWithPrefix(space, kGpuPlanePrefix);
  if (!gpu_planes.empty()) {
    absl::string_view gpu_model =
        GpuModelName(GetDeviceCaps(*gpu_planes.front()));
    if (!gpu_model.empty()) {
      env->set_device_type(std::string(gpu_model));
    } else {
      env->set_device_type("GPU");
    }
    env->set_device_core_count(gpu_planes.size());
    env->set_hardware_type(tensorflow::profiler::HardwareType::GPU);
  } else if (std::vector<const XPlane*> tpu_planes =
                 FindTensorCorePlanes(space);
             !tpu_planes.empty()) {
    XPlaneVisitor visitor =
        tsl::profiler::CreateTfXPlaneVisitor(tpu_planes.at(0));
    auto xstat = visitor.GetStat(StatType::kDeviceTypeString);
    if (xstat.has_value()) {
      env->set_device_type(std::string(xstat->StrOrRefValue()));
    }
    env->set_device_core_count(tpu_planes.size());
    env->set_hardware_type(tensorflow::profiler::HardwareType::TPU);
  } else {
    env->set_device_type("CPU");
    env->set_device_core_count(0);
    env->set_hardware_type(tensorflow::profiler::HardwareType::CPU_ONLY);
  }
}

void PropagateXSpaceDiagnosticsToOpStats(const XSpace& space,
                                         OpStats* op_stats) {
  if (!space.errors().empty()) {
    absl::flat_hash_set<std::string> unique_errors;
    unique_errors.insert(space.errors().begin(), space.errors().end());
    *op_stats->mutable_diagnostics()->mutable_errors() = {unique_errors.begin(),
                                                          unique_errors.end()};
  }
  if (!space.warnings().empty()) {
    absl::flat_hash_set<std::string> unique_warnings;
    unique_warnings.insert(space.warnings().begin(), space.warnings().end());
    *op_stats->mutable_diagnostics()->mutable_warnings() = {
        unique_warnings.begin(), unique_warnings.end()};
  }
}

// This function should be idempotent to be called
void SetProgramIdToNameMap(const HloProtoMap& hlo_proto_map,
                           tensorflow::profiler::OpStats& op_stats) {
  auto& program_id_to_name_map = *op_stats.mutable_program_id_to_name_map();
  for (const auto& [program_id, hlo_proto] : hlo_proto_map) {
    program_id_to_name_map[program_id] = hlo_proto->hlo_module().name();
  }
}

void UpdateOpMetricsDbFromHloModuleMap(OpMetricsDb& op_metrics_db,
                                       const HloModuleMap& hlo_module_map) {
  for (OpMetrics& op_metrics : *op_metrics_db.mutable_metrics_db()) {
    EnterOpMetadataFromHloModuleMap(&op_metrics, hlo_module_map);
  }
}

DutyCycleTracker ConstructDutyCycleTracker(XPlaneVisitor& visitor) {
  DutyCycleTracker duty_cycle_tracker;
  visitor.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() == tsl::profiler::kXlaOpLineName) {
      line.ForEachEvent([&](const XEventVisitor& event) {
        auto hlo_category_stat = event.GetStat(StatType::kHloCategory);
        duty_cycle_tracker.AddInterval(
            event.GetTimespan(),
            !(hlo_category_stat &&
              tsl::profiler::IsOffDutyOp(hlo_category_stat->StrOrRefValue())));
      });
    } else if (line.Name() == tsl::profiler::kSparseCoreOpLineName) {
      line.ForEachEvent([&](const XEventVisitor& event) {
        //  TODO(b/397774568): Add support for SC off-duty ops.
        duty_cycle_tracker.AddInterval(event.GetTimespan(), /*is_active=*/true);
      });
    } else if (line.Name() == tsl::profiler::kXlaModuleLineName ||
               line.Name() == tsl::profiler::kSparseCoreModuleLineName) {
      line.ForEachEvent([&](const XEventVisitor& event) {
        duty_cycle_tracker.AddInterval(event.GetTimespan(),
                                       /*is_active=*/false);
      });
    }
  });
  return duty_cycle_tracker;
}

OpStats ConvertXSpaceToOpStats(const XSpace& space,
                               const OpStatsOptions& options) {
  OpStats op_stats;
  StepEvents step_events;
  PropagateXSpaceDiagnosticsToOpStats(space, &op_stats);
  // Convert device planes.
  OpMetricsDbCombiner op_metrics_db_combiner(
      op_stats.mutable_device_op_metrics_db());
  SetRunEnvironment(space, op_stats.mutable_run_environment());

  KernelReportMap reports;

  // Handle device planes first. device_planes will contain either GPU or TPU.
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(space, kTpuPlanePrefix);
  const bool is_gpu = device_planes.empty();
  if (is_gpu) {
    device_planes = FindPlanesWithPrefix(space, kGpuPlanePrefix);
  }
  const bool is_tpu = !is_gpu;
  std::string hostname = Hostname(space);
  auto& core_id_to_details_map = *op_stats.mutable_core_id_to_details();
  if (is_gpu) {
    core_id_to_details_map[kDefaultGpuLocalCoreId].set_hostname(hostname);
  }
  DutyCycleCombiner duty_cycle_combiner;
  // TODO(b/161942993) parallelize XPlane processing per thread.
  HloModuleMap hlo_module_map;

  // Generate HloModuleMap if kernel stats or op metrics for TPU are requested.
  bool generate_hlo_module_map = options.generate_kernel_stats_db ||
                                 (is_tpu && options.generate_op_metrics_db);
  if (generate_hlo_module_map) {
    tensorflow::profiler::HloCostAnalysisWrapper::Factory create_cost_analysis;
    if (is_gpu) {
      create_cost_analysis = []() {
        return GetHloCostAnalysisWrapperRegistry().Get(
            kXprofGpuCostAnalysisName)(nullptr);};
    } else {
      // we pass nullptr for the cost analysis for TPU.
      create_cost_analysis = []() { return nullptr; };
    }
    ProcessHloModuleMapFromXSpace(hlo_module_map, &space, create_cost_analysis);
  }
  {
    auto executor =
        std::make_unique<XprofThreadPoolExecutor>("op_stats_threads");

    // OpMetricDb Generation.
    std::vector<OpMetricsDb> all_op_metrics_dbs;

    // Ensure op_metrics threads are joined and results combined when the
    // function exits.
    auto op_metrics_cleanup =
        absl::MakeCleanup([&all_op_metrics_dbs, &op_metrics_db_combiner]() {
          for (auto& op_metrics_db : all_op_metrics_dbs) {
            op_metrics_db_combiner.Combine(op_metrics_db);
          }
        });

    if (options.generate_op_metrics_db) {
      all_op_metrics_dbs.resize(device_planes.size());  // Resize here

      if (!device_planes.empty() && !op_stats.has_perf_env()) {
        *op_stats.mutable_perf_env() = GetPerfEnvFromXPlane(*device_planes[0]);
      }
      for (size_t i = 0; i < device_planes.size(); ++i) {
        const XPlane* device_plane = device_planes[i];
        OpMetricsDb& op_metrics_db = all_op_metrics_dbs[i];
        executor->Execute([device_plane, &hlo_module_map, is_tpu,
                           &op_metrics_db]() {
          if (!is_tpu) {
            op_metrics_db = ConvertDeviceTraceXPlaneToOpMetricsDb(
                *device_plane, hlo_module_map);
          } else {
            // TODO(b/397774568): Remove this once the SparseCore
            // OpMetricsDb is implemented.
            if (!tsl::profiler::GetSparseCoreId(device_plane->name())
                     .has_value()) {
              op_metrics_db =
                  ConvertTpuDeviceTraceXPlaneToOpMetricsDb(*device_plane);
              UpdateOpMetricsDbFromHloModuleMap(op_metrics_db, hlo_module_map);
            }
          }
        });
      }
    }

    // StepDb Generation.
    std::vector<StepEvents> all_step_events;

    // Ensure step_events threads are joined and results combined when the
    // function exits.
    auto step_events_cleanup =
        absl::MakeCleanup([&all_step_events, &step_events, is_tpu]() {
          for (auto& device_step_events : all_step_events) {
            if (is_tpu) {
              // In TPU, we take the intersection of step events across cores
              // as well as hosts.see b/158249775 and cl/331842545.
              IntersectCombineStepEvents(device_step_events, &step_events);
            } else {
              UnionCombineStepEvents(device_step_events, &step_events);
            }
          }
        });
    if (options.generate_step_db) {
      all_step_events.resize(device_planes.size());
      for (size_t i = 0; i < device_planes.size(); ++i) {
        const XPlane* device_trace = device_planes[i];
        auto& current_step_events = all_step_events[i];
        executor->Execute([device_trace, &current_step_events]() {
          current_step_events =
              ConvertDeviceTraceXPlaneToStepEvents(*device_trace);
        });
      }
    }
    std::vector<KernelReportMap> kernel_reports;
    // Ensure step_events threads are joined and results combined when the
    // function exits.
    auto kernel_reports_cleanup =
        absl::MakeCleanup([&kernel_reports, &reports]() {
          for (auto& kernel_report : kernel_reports) {
            for (auto& kernel_report_entry : kernel_report) {
              InsertOrUpdateKernelReport(kernel_report_entry.first,
                                         kernel_report_entry.second, &reports);
            }
          }
        });
    if (options.generate_kernel_stats_db) {
      kernel_reports.resize(device_planes.size());
      for (size_t i = 0; i < device_planes.size(); ++i) {
        const XPlane* device_trace = device_planes[i];
        KernelReportMap& current_report = kernel_reports[i];
        executor->Execute([device_trace, &hlo_module_map, &current_report]() {
          ConvertDeviceTraceXPlaneToKernelReports(
              *device_trace,
              // TODO(cleanup): Move this to xplane_to_kernel_stats_db.cc
              [&](const GpuEventStats& stats, KernelReport* kernel) {
                if (!stats.IsXlaOp()) return;
                const HloInstructionWrapper* hlo_instruction =
                    GetHloInstruction(hlo_module_map, stats.program_id,
                                      stats.hlo_op_names.back());
                if (hlo_instruction != nullptr) {
                  kernel->set_op_name(std::string(hlo_instruction->TfOpName()));
                  bool tc_eligible = IsOpTensorCoreEligible(kernel->op_name());
                  if (VLOG_IS_ON(1) && !tc_eligible &&
                      kernel->is_kernel_using_tensor_core()) {
                    VLOG(1) << "Detected new Op using TensorCores: "
                            << kernel->op_name() << std::endl;
                  }
                  kernel->set_is_op_tensor_core_eligible(
                      tc_eligible || kernel->is_op_tensor_core_eligible());
                }
              },
              &current_report);  // Write to the thread-local report map
        });
      }
    }

    // Device Trace generation.
    struct DeviceTraceResult {
      DutyCycleTracker duty_cycle_tracker;
      std::optional<CoreDetails> core_details;
    };
    std::vector<DeviceTraceResult> device_trace_results;

    // Ensure device_trace threads are joined and results processed when the
    // function exits.
    auto device_trace_cleanup =
        absl::MakeCleanup([&device_trace_results, &device_planes,
                           &core_id_to_details_map, &duty_cycle_combiner]() {
          for (size_t i = 0; i < device_planes.size(); ++i) {
            const XPlane* device_trace = device_planes[i];
            const auto& result = device_trace_results[i];
            if (result.core_details.has_value()) {
              core_id_to_details_map[device_trace->id()] = *result.core_details;
              duty_cycle_combiner.CombineCore(
                  result.duty_cycle_tracker,
                  result.core_details->local_chip_id());
            } else {
              LOG(WARNING) << "No CoreDetails found for TPU device plane: "
                           << device_trace->name();
              duty_cycle_combiner.CombineChip(result.duty_cycle_tracker);
            }
          }
        });
    device_trace_results.resize(device_planes.size());
    for (size_t i = 0; i < device_planes.size(); ++i) {
      const XPlane* device_trace = device_planes[i];
      auto& device_trace_result = device_trace_results[i];
      executor->Execute([device_trace, &hostname, &device_trace_result]() {
        XPlaneVisitor visitor =
            tsl::profiler::CreateTfXPlaneVisitor(device_trace);
        DutyCycleTracker duty_cycle_tracker =
            ConstructDutyCycleTracker(visitor);
        std::optional<CoreDetails> core_details;
        if (std::optional<XStatVisitor> core_details_stat =
                visitor.GetStat(StatType::kCoreDetails)) {
          core_details.emplace();
          absl::string_view core_details_bytes =
              core_details_stat->BytesValue();
          if (core_details->ParseFromArray(core_details_bytes.data(),
                                           core_details_bytes.size())) {
            core_details->set_hostname(hostname);
            core_details->set_is_sparse_core(
                tsl::profiler::GetSparseCoreId(device_trace->name())
                    .has_value());
          } else {
            core_details.reset();
          }
        }
        device_trace_result = {duty_cycle_tracker, core_details};
      });
    }
    // All event generation should end in this block before we start combining
    executor->JoinAll();  // Wait for all scheduled tasks to complete.
                          // The cleanup blocks will execute after this step.
  }

  for (const auto& [program_id, hlo_module] : hlo_module_map) {
    ModelTracker model_tracker;
    model_tracker.ProcessHloModule(hlo_module);
    if (model_tracker.IsTraining()) {
      op_stats.mutable_run_environment()->set_is_training(true);
      break;
    }
  }

  // Start combining data.
  if (is_tpu) {
    OpMetricsDb& op_metrics_db = *op_stats.mutable_device_op_metrics_db();
    op_metrics_db.set_idle_time_ps(duty_cycle_combiner.GetTotalIdleTimePs());
    op_metrics_db.set_busy_time_ps(duty_cycle_combiner.GetTotalActiveTimePs());
  }

  // Combine into reports.
  if (options.generate_kernel_stats_db) {
    CopyTopKDurationKernelReportsToDb(reports,
                                      op_stats.mutable_kernel_stats_db());
  }

  bool has_device = !device_planes.empty();
  // Convert a host plane.
  const XPlane* host_plane = tsl::profiler::FindPlaneWithName(
      space, tsl::profiler::kHostThreadsPlaneName);
  if (host_plane) {
    if (options.generate_op_metrics_db) {
      *op_stats.mutable_host_op_metrics_db() =
          ConvertHostThreadsXPlaneToOpMetricsDb(*host_plane);
    }
    if (options.generate_step_db && !has_device) {
      StepEvents host_step_events =
          ConvertHostThreadsXPlaneToStepEvents(*host_plane, nullptr);
      UnionCombineStepEvents(host_step_events, &step_events);
    }
    XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(host_plane);
    auto stat = visitor.GetStat(StatType::kMatrixUnitUtilizationPercent);
    if (stat.has_value()) {
      op_stats.mutable_performance_counter_result()
          ->set_matrix_unit_utilization_percent(stat->DoubleValue());
    }
    TfFunctionDb* tf_function_db = op_stats.mutable_tf_function_db();
    visitor.ForEachLine([&](const XLineVisitor& line) {
      CombineTfFunctionDb(ConvertHostThreadsXLineToTfFunctionDb(line),
                          tf_function_db);
    });
  }
  if (options.generate_step_db) {
    if (is_tpu) {
      // TPU steps relies on step number in step line in Xplane which has
      // already dropped the incomplete steps at both beginning and end.
      *op_stats.mutable_step_db() = ConvertStepEventsToStepDb(
          has_device, /*maybe_drop_incomplete_steps=*/false, step_events);
      *op_stats.mutable_device_op_metrics_db()->mutable_precision_stats() =
          ComputePrecisionStats(step_events);
      OpMetricsDbCombiner combiner(
          op_stats.mutable_hlo_metrics_db_complete_steps_only());
      for (const auto& step_info : op_stats.step_db().step_sequence()) {
        combiner.Combine(step_info.hlo_metrics_db());
      }
    } else {
      StepEvents nonoverlapped_step_events =
          ToNonOverlappedStepEvents(step_events);
      *op_stats.mutable_step_db() = ConvertStepEventsToStepDb(
          has_device, options.maybe_drop_incomplete_steps,
          nonoverlapped_step_events);
      *op_stats.mutable_device_op_metrics_db()->mutable_precision_stats() =
          ComputePrecisionStats(nonoverlapped_step_events);
    }
  }

  // Set program_id_to_name map in OpStats from Xspace
  // Will be non-op if the space does not have materialized device traces
  HloProtoMap hlo_proto_map;
  hlo_proto_map.AddHloProtosFromXSpace(space);
  SetProgramIdToNameMap(hlo_proto_map, op_stats);

  return op_stats;
}

}  // namespace profiler
}  // namespace tensorflow
