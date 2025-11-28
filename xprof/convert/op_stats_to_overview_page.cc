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

#include "xprof/convert/op_stats_to_overview_page.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/format_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/op_metrics_to_record.h"
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/kernel_stats.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "plugin/xprof/protobuf/power_metrics.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "plugin/xprof/protobuf/tf_function.pb.h"
#include "xprof/utils/diagnostics.h"
#include "xprof/utils/hardware_type_utils.h"
#include "xprof/utils/html_utils.h"
#include "xprof/utils/kernel_stats_utils.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

using tsl::profiler::OneDigit;

// If the use of low-precision ops is less than this percentage threshold, a
// statement of suggestion will be made.
constexpr double kLowPrecisionPercentThreshold = 10;

struct TfFunctionInfo {
  absl::string_view function_name;
  double expensive_call_percent;
};

OverviewPageTip MakeOverviewPageTip(std::string text) {
  OverviewPageTip tip;
  tip.set_link(std::move(text));
  return tip;
}

// Makes a recommendation for looking up a document.
// doc_url is expected to be already be escaped suitably for use in an HTML
// attribute.
OverviewPageTip MakeOverviewPageTipDocLink(absl::string_view doc_url,
                                           absl::string_view text) {
  return MakeOverviewPageTip(AnchorElement(doc_url, text));
}

void ComputeHostTips(OverviewPageRecommendation* re) {
  *re->add_host_tips() = MakeOverviewPageTip(
      "input_pipeline_analyzer (especially Section 3 for the breakdown of "
      "input operations on the Host)");
  *re->add_host_tips() = MakeOverviewPageTip(
      "trace_viewer (look at the activities on the timeline of each Host "
      "Thread near the bottom of the trace view)");
}

void ComputeDeviceTips(HardwareType hardware_type,
                       OverviewPageRecommendation* re) {
  absl::string_view device_name = HardwareType_Name(hardware_type);
  absl::string_view timeline_name = device_name;
  absl::string_view op_stats_toolname = "framework_op_stats";
  if (hardware_type == tensorflow::profiler::TPU) {
    timeline_name = "TPU core";
    op_stats_toolname = "op_profile";
  }
  *re->add_device_tips() = MakeOverviewPageTip(
      absl::StrCat(op_stats_toolname,
                   " (identify the time-consuming operations "
                   "executed on the ",
                   device_name, ")"));
  *re->add_device_tips() = MakeOverviewPageTip(absl::StrCat(
      "trace_viewer (look at the activities on the timeline of each ",
      timeline_name, " in the trace view)"));
}

void ComputeFaqTips(OverviewPageRecommendation* re) {
  *re->add_faq_tips() = MakeOverviewPageTip("Refer to the TF2 Profiler FAQ");
}

void ComputeDocumentationTips(OverviewPageRecommendation* re) {
  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/data_performance_analysis",
      "Analyze tf.data performance with the TF Profiler");
  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/"
      "data_performance",
      "Better performance with the tf.data API");
}

std::string GeneratePrecisionStatement(const PrecisionStats& precision_stats) {
  uint64_t total_compute_ps =
      precision_stats.compute_16bit_ps() + precision_stats.compute_32bit_ps();
  if (total_compute_ps > 0) {
    double percent_16bit =
        (100.0 * precision_stats.compute_16bit_ps()) / total_compute_ps;
    if (percent_16bit < kLowPrecisionPercentThreshold) {
      return absl::StrCat(
          "Only ", OneDigit(percent_16bit),
          "% of device computation is 16 bit. So you might want to replace "
          "more 32-bit Ops by 16-bit Ops to improve performance (if the "
          "reduced accuracy is acceptable).");
    }
  }
  return "";
}

}  // namespace

void SetCommonRecommendation(
    absl::string_view input_classification, absl::string_view input_statement,
    absl::string_view output_statement, HardwareType hardware_type,
    absl::string_view tf_function_statement_html,
    absl::string_view eager_statement_html,
    absl::string_view outside_compilation_statement_html,
    OverviewPageRecommendation* re) {
  re->set_bottleneck(std::string(input_classification));
  re->set_statement(std::string(input_statement));
  re->set_output_statement(std::string(output_statement));
  re->set_tf_function_statement_html(std::string(tf_function_statement_html));
  re->set_eager_statement_html(std::string(eager_statement_html));
  re->set_outside_compilation_statement_html(
      std::string(outside_compilation_statement_html));
  ComputeHostTips(re);
  ComputeDeviceTips(hardware_type, re);
  ComputeDocumentationTips(re);
  ComputeFaqTips(re);
}

OverviewPageRecommendation ComputeGenericRecommendation(
    const BottleneckAnalysis& bottleneck,
    const PrecisionStats& precision_stats) {
  OverviewPageRecommendation re;
  GenericRecommendation generic;
  generic.set_device_collectives_bottleneck(
      bottleneck.device_collectives_classification());
  generic.set_device_collectives_statement(
      bottleneck.device_collectives_statement());
  generic.set_kernel_launch_bottleneck(
      bottleneck.kernel_launch_classification());
  generic.set_kernel_launch_statement(bottleneck.kernel_launch_statement());
  generic.set_all_other_bottleneck(bottleneck.all_other_classification());
  generic.set_all_other_statement(bottleneck.all_other_statement());
  generic.set_precision_statement(GeneratePrecisionStatement(precision_stats));
  re.mutable_recommendation()->PackFrom(generic);
  return re;
}

OverviewPageAnalysis ComputeAnalysisResult(const OpStats& op_stats) {
  OverviewPageAnalysis analysis;
  OpMetricsDb device_tf_op_metrics_db = CreateTfMetricsDbFromDeviceOpMetricsDb(
      op_stats.device_op_metrics_db(), /*with_idle=*/false);
  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  uint64_t total_device_time_ps = device_tf_op_metrics_db.total_time_ps();
  constexpr int kNumTopOpsShown = 10;
  double device_cumulative_fraction = 0.0;
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(device_tf_op_metrics_db, kNumTopOpsShown)) {
    OverviewTfOp* op = analysis.add_top_device_ops();
    op->set_name(metrics->name());
    op->set_category(metrics->category());
    op->set_self_time_fraction(tsl::profiler::SafeDivide(
        metrics->self_time_ps(), total_device_time_ps));
    device_cumulative_fraction += op->self_time_fraction();
    op->set_cumulative_time_fraction(device_cumulative_fraction);
    op->set_flop_rate(tsl::profiler::SafeDivide(
        metrics->flops(), tsl::profiler::PicoToNano(metrics->time_ps())));
    auto iter = kernel_stats_by_op_name.find(op->name());
    if (iter != kernel_stats_by_op_name.end()) {
      op->set_is_op_tensorcore_eligible(
          iter->second.is_op_tensor_core_eligible);
      op->set_is_op_using_tensorcore(iter->second.tensor_core_duration_ns != 0);
    }
  }
  uint64_t total_device_compute_ps =
      op_stats.device_op_metrics_db().precision_stats().compute_16bit_ps() +
      op_stats.device_op_metrics_db().precision_stats().compute_32bit_ps();
  analysis.set_device_compute_16bit_percent(
      100.0 *
      tsl::profiler::SafeDivide(
          op_stats.device_op_metrics_db().precision_stats().compute_16bit_ps(),
          total_device_compute_ps));
  analysis.set_device_compute_32bit_percent(
      100.0 *
      tsl::profiler::SafeDivide(
          op_stats.device_op_metrics_db().precision_stats().compute_32bit_ps(),
          total_device_compute_ps));

  uint64_t num_host_tf_ops = 0;
  uint64_t total_host_op_time_ps_exclude_idle = 0;
  uint64_t eager_host_op_time_ps = 0;
  for (const OpMetrics& metrics : op_stats.host_op_metrics_db().metrics_db()) {
    num_host_tf_ops += metrics.occurrences();
    if (!IsIdleOp(metrics)) {
      total_host_op_time_ps_exclude_idle += metrics.self_time_ps();
      if (metrics.is_eager()) eager_host_op_time_ps += metrics.self_time_ps();
    }
  }
  uint64_t num_device_tf_ops = 0;
  uint64_t total_device_op_time_ps_exclude_idle = 0;
  uint64_t eager_device_op_time_ps = 0;
  for (const OpMetrics& metrics : device_tf_op_metrics_db.metrics_db()) {
    num_device_tf_ops += metrics.occurrences();
    if (!IsIdleOp(metrics)) {
      total_device_op_time_ps_exclude_idle += metrics.self_time_ps();
      if (metrics.is_eager()) eager_device_op_time_ps += metrics.self_time_ps();
    }
  }
  // Figures out outside_compilation time from
  // op_stats.device_op_metrics_db().metrics_db(). We don't use the
  // {metrics.provenance(), metrics.name()} from
  // device_tf_op_metrics_db.metrics_db(), because metrics.provenance() there is
  // not set and metrics.name() can be either HLO-Op name or TF-Op name, which
  // will confuse tsl::profiler::IsOutsideCompilationOp().
  uint64_t outside_compilation_device_op_time_ps = 0;
  for (const OpMetrics& metrics :
       op_stats.device_op_metrics_db().metrics_db()) {
    if (!tsl::profiler::IsOutsideCompilationOp(metrics.provenance(),
                                               metrics.long_name()))
      continue;
    outside_compilation_device_op_time_ps += metrics.self_time_ps();
  }
  uint64_t num_total_tf_ops = num_host_tf_ops + num_device_tf_ops;
  analysis.set_host_tf_op_percent(
      100.0 * tsl::profiler::SafeDivide(num_host_tf_ops, num_total_tf_ops));
  analysis.set_device_tf_op_percent(
      100.0 * tsl::profiler::SafeDivide(num_device_tf_ops, num_total_tf_ops));
  analysis.set_host_trace_level(op_stats.run_environment().host_trace_level());
  analysis.set_host_op_time_eager_percent(
      100.0 * tsl::profiler::SafeDivide(eager_host_op_time_ps,
                                        total_host_op_time_ps_exclude_idle));
  analysis.set_device_op_time_eager_percent(
      100.0 * tsl::profiler::SafeDivide(eager_device_op_time_ps,
                                        total_device_op_time_ps_exclude_idle));
  analysis.set_device_op_time_outside_compilation_percent(
      100.0 * tsl::profiler::SafeDivide(outside_compilation_device_op_time_ps,
                                        total_device_op_time_ps_exclude_idle));
  return analysis;
}

// Converts from HostIndependentJobInfo to OverviewPageHostIndependentJobInfo.
OverviewPageHostIndependentJobInfo ToOverviewPageHostIndependentJobInfo(
    const HostIndependentJobInfoResult& host_independent_job_info) {
  OverviewPageHostIndependentJobInfo result;
  result.set_change_list(host_independent_job_info.change_list());
  result.set_workspace_id(host_independent_job_info.workspace_id());
  result.set_snapshot(host_independent_job_info.snapshot());
  result.set_build_time(host_independent_job_info.build_time());
  result.set_build_target(host_independent_job_info.build_target());
  result.set_profile_duration_ms(
      host_independent_job_info.profile_duration_ms());
  return result;
}

// Converts from HostDependentJobInfo to OverviewPageHostDependentJobInfo.
OverviewPageHostDependentJobInfo ToOverviewPageHostDependentJobInfo(
    const HostDependentJobInfoResult& host_dependent_job_info) {
  OverviewPageHostDependentJobInfo result;
  result.set_host_id(host_dependent_job_info.host_id());
  result.set_command_line(host_dependent_job_info.command_line());
  result.set_start_time(host_dependent_job_info.start_time());
  result.set_bns_address(host_dependent_job_info.bns_address());
  result.set_profile_time_ns(host_dependent_job_info.profile_time_ns());
  return result;
}

OverviewPageRunEnvironment ComputeRunEnvironment(
    const RunEnvironment& run_environment) {
  OverviewPageRunEnvironment re;
  re.set_host_count(run_environment.host_count());
  re.set_task_count(run_environment.task_count());
  re.set_device_type(run_environment.device_type());
  re.set_device_core_count(run_environment.device_core_count());
  re.set_replica_count(run_environment.replica_count());
  re.set_num_cores_per_replica(run_environment.num_cores_per_replica());
  re.set_is_training(run_environment.is_training());
  if (run_environment.has_power_metrics()) {
    *re.mutable_power_metrics() = run_environment.power_metrics();
  }
  *re.mutable_host_independent_job_info() =
      ToOverviewPageHostIndependentJobInfo(
          run_environment.host_independent_job_info());
  for (const auto& host_dependent_job_info :
       run_environment.host_dependent_job_info()) {
    *re.add_host_dependent_job_info() =
        ToOverviewPageHostDependentJobInfo(host_dependent_job_info);
  }
  return re;
}

std::string TfFunctionRecommendationHtml(const TfFunctionDb& tf_function_db) {
  std::vector<TfFunctionInfo> candidates;
  for (const auto& name_fun : tf_function_db.tf_functions()) {
    const auto& fun = name_fun.second;
    if (fun.expensive_call_percent() >= kTfFunctionReportThresholdInPercent) {
      candidates.push_back({name_fun.first, fun.expensive_call_percent()});
    }
  }
  if (candidates.empty()) return "";
  auto cmp = [](const TfFunctionInfo& a, const TfFunctionInfo& b) {
    return a.expensive_call_percent > b.expensive_call_percent;
  };
  // Sorts candidates in descending order of expensive_call_percent.
  absl::c_sort(candidates, cmp);
  std::string expensive_functions = "";
  auto num_functions_shown = std::min(
      static_cast<decltype(candidates)::size_type>(3), candidates.size());

  for (decltype(candidates)::size_type i = 0; i < num_functions_shown; i++) {
    if (i > 0) absl::StrAppend(&expensive_functions, ", ");
    absl::StrAppend(&expensive_functions, "\"", candidates[i].function_name,
                    "\"");
  }
  if (candidates.size() > num_functions_shown)
    absl::StrAppend(&expensive_functions, " and more");
  return absl::StrCat("Expensive tf-functions detected (", expensive_functions,
                      ") due to either retracing or eager execution.");
}

std::string EagerRecommendationHtml(double host_op_time_eager_percent,
                                    double device_op_time_eager_percent) {
  std::string recommendation = "";
  if (host_op_time_eager_percent > kEagerReportThresholdInPercent)
    absl::StrAppend(&recommendation, OneDigit(host_op_time_eager_percent),
                    "% of Op time on the host used eager execution. ");
  if (device_op_time_eager_percent > kEagerReportThresholdInPercent)
    absl::StrAppend(&recommendation, OneDigit(device_op_time_eager_percent),
                    "% of Op time on the device used eager execution. ");
  if (!recommendation.empty())
    absl::StrAppend(&recommendation, "Performance could be improved with ",
                    AnchorElement("https://www.tensorflow.org/guide/function",
                                  "tf.function."));
  return recommendation;
}

std::string OutsideCompilationRecommendationHtml(
    double device_op_time_outside_compilation_percent) {
  if (device_op_time_outside_compilation_percent <=
      kOutsideCompilationThresholdInPercent)
    return "";
  return absl::StrCat(
      OneDigit(device_op_time_outside_compilation_percent),
      " % of Op time on the device are for outside compilation. Performance "
      "could be improved by avoiding outside compilation.");
}

OverviewPage ConvertOpStatsToOverviewPage(const OpStats& op_stats) {
  OverviewPage overview_page;
  *overview_page.mutable_run_environment() =
      ComputeRunEnvironment(op_stats.run_environment());
  *overview_page.mutable_analysis() = ComputeAnalysisResult(op_stats);
  *overview_page.mutable_input_analysis() =
      ConvertOpStatsToInputPipelineAnalysis(op_stats);
  BottleneckAnalysis bottleneck = ComputeBottleneckAnalysis(
      overview_page.input_analysis().input_time_breakdown(),
      overview_page.input_analysis().step_details());
  *overview_page.mutable_recommendation() = ComputeGenericRecommendation(
      bottleneck, op_stats.device_op_metrics_db().precision_stats());
  SetCommonRecommendation(
      bottleneck.input_classification(), bottleneck.input_statement(), "",
      ParseHardwareType(op_stats.run_environment().device_type()),
      TfFunctionRecommendationHtml(op_stats.tf_function_db()),
      EagerRecommendationHtml(
          overview_page.analysis().host_op_time_eager_percent(),
          overview_page.analysis().device_op_time_eager_percent()),
      OutsideCompilationRecommendationHtml(
          overview_page.analysis()
              .device_op_time_outside_compilation_percent()),
      overview_page.mutable_recommendation());
  PopulateOverviewDiagnostics(op_stats, overview_page.mutable_diagnostics());
  overview_page.mutable_analysis()->set_mxu_utilization_percent(
      op_stats.performance_counter_result().matrix_unit_utilization_percent());
  overview_page.mutable_analysis()->set_hbm_utilization_percent(
      op_stats.performance_counter_result().hbm_utilization_percent());
  return overview_page;
}

std::string StrFormatToPercentage(double num) {
  return (num >= 0) ? absl::StrFormat("%.1lf%%", num) : "unknown";
}

void AddCommonStats(DataTable* data_table,
                    const OverviewPageAnalysis& analysis) {
  data_table->AddCustomProperty("host_trace_level",
                                absl::StrCat(analysis.host_trace_level()));
  data_table->AddCustomProperty(
      "host_tf_op_percent",
      StrFormatToPercentage(analysis.host_tf_op_percent()));
  data_table->AddCustomProperty(
      "device_tf_op_percent",
      StrFormatToPercentage(analysis.device_tf_op_percent()));
  data_table->AddCustomProperty(
      "host_op_time_eager_percent",
      StrFormatToPercentage(analysis.host_op_time_eager_percent()));
  data_table->AddCustomProperty(
      "device_op_time_eager_percent",
      StrFormatToPercentage(analysis.device_op_time_eager_percent()));
  data_table->AddCustomProperty("remark_text", analysis.remark_text());
  data_table->AddCustomProperty("remark_color", analysis.remark_color());
  data_table->AddCustomProperty(
      "mxu_utilization_percent",
      StrFormatToPercentage(analysis.mxu_utilization_percent()));
  data_table->AddCustomProperty(
      "hbm_utilization_percent",
      StrFormatToPercentage(analysis.hbm_utilization_percent()));
  if (analysis.program_goodput_percent()) {
    data_table->AddCustomProperty(
        "program_goodput_percent",
        StrFormatToPercentage(analysis.program_goodput_percent()));
  }
  if (analysis.sc_step_time_ms_average()) {
    data_table->AddCustomProperty(
        "sc_steptime_average",
        absl::StrFormat("%.1f", analysis.sc_step_time_ms_average()));
    data_table->AddCustomProperty(
        "sc_infeed_time_avg",
        absl::StrFormat("%.1f", analysis.sc_infeed_time_ms_avg()));
    data_table->AddCustomProperty(
        "sc_outfeed_time_avg",
        absl::StrFormat("%.1f", analysis.sc_outfeed_time_ms_avg()));
    data_table->AddCustomProperty(
        "sc_idle_time_avg",
        absl::StrFormat("%.1f", analysis.sc_idle_time_ms_avg()));
  }
}

std::unique_ptr<DataTable> GenerateRunEnvironmentDataTable(
    const OverviewPage& result) {
  auto run_environment_data_table = std::make_unique<DataTable>();

  // For each host, adds one row to the table.
  std::vector<std::vector<std::string>> kColumns = {
      {"host_id", "string", "host_id"},
      {"command_line", "string", "command_line"},
      {"start_time", "string", "start_time"},
      {"bns_address", "string", "bns_address"}};
  for (const auto& column : kColumns) {
    run_environment_data_table->AddColumn(
        TableColumn(column[0], column[1], column[2]));
  }

  const auto& run_environment = result.run_environment();
  const auto& host_dependent_job_info =
      run_environment.host_dependent_job_info();

  for (const auto& host_dependent_job : host_dependent_job_info) {
    TableRow* row = run_environment_data_table->AddRow();
    row->AddTextCell(host_dependent_job.host_id());
    row->AddTextCell(host_dependent_job.command_line());
    row->AddTextCell(
        absl::FormatTime(absl::FromUnixNanos(host_dependent_job.start_time()),
                         absl::UTCTimeZone()));
    row->AddTextCell(host_dependent_job.bns_address());
  }

  const std::string host_count =
      (run_environment.host_count() >= 0)
          ? absl::StrCat(run_environment.host_count())
          : "unknown";
  run_environment_data_table->AddCustomProperty("host_count", host_count);

  std::string task_count = (run_environment.task_count() >= 0)
                               ? absl::StrCat(run_environment.task_count())
                               : "unknown";
  if (run_environment.task_count() > 0 && run_environment.host_count() > 0) {
    int32_t tasks_per_host =
        run_environment.task_count() / run_environment.host_count();
    absl::StrAppend(&task_count, " (num tasks per host = ", tasks_per_host,
                    ")");
  }
  run_environment_data_table->AddCustomProperty("task_count", task_count);

  run_environment_data_table->AddCustomProperty("device_type",
                                                run_environment.device_type());
  std::string device_core_count =
      (run_environment.device_core_count() >= 0)
          ? absl::StrCat(run_environment.device_core_count())
          : "unknown";
  if (run_environment.replica_count() > 0 &&
      run_environment.num_cores_per_replica() > 0) {
    absl::StrAppend(&device_core_count,
                    " (Replica count = ", run_environment.replica_count(),
                    ", num cores per replica = ",
                    run_environment.num_cores_per_replica(), ")");
  }
  run_environment_data_table->AddCustomProperty("device_core_count",
                                                device_core_count);
  run_environment_data_table->AddCustomProperty(
      "is_training", absl::StrFormat("%v", run_environment.is_training()));

  return run_environment_data_table;
}

void AddLatencyRow(DataTable* data_table, absl::string_view label,
                   const OverviewLatencyBreakdown& breakdown) {
  TableRow* row = data_table->AddRow();
  row->AddTextCell(label);
  row->AddNumberCell(tsl::profiler::MicroToMilli(breakdown.host_latency_us()));
  row->AddNumberCell(
      tsl::profiler::MicroToMilli(breakdown.device_latency_us()));
  row->AddNumberCell(
      tsl::profiler::MicroToMilli(breakdown.communication_latency_us()));
  row->AddNumberCell(tsl::profiler::MicroToMilli(breakdown.total_latency_us()));
}

// Converts an OverViewInferenceLatency proto to a DataTable
// (which has "rows x columns" of data items plus some table properties).
// #rows = #percentile numbers, #columns = #breakdown items.
std::unique_ptr<DataTable> GenerateInferenceLatencyDataTable(
    const OverviewInferenceLatency& result) {
  std::vector<std::vector<std::string>> kColumns = {
      {"percentile", "string", "percentile"},
      {"hostTimeMs", "number", "Host time (in ms)"},
      {"deviceTimeMs", "number", "Device time (in ms)"},
      {"communicationTimeMs", "number",
       "Host-device communication time (in ms)"},
      {"totalTimeMs", "number", "Total latency (in ms)"}};

  auto data_table = std::make_unique<DataTable>();
  for (const auto& column : kColumns) {
    data_table->AddColumn(TableColumn(column[0], column[1], column[2]));
  }

  if (result.latency_breakdowns_size() > 0) {
    AddLatencyRow(data_table.get(), "Avg",
                  *result.latency_breakdowns().begin());
  }
  for (int i = 0; i < result.percentile_numbers_size(); i++) {
    AddLatencyRow(
        data_table.get(),
        absl::StrFormat("%.1f%%", (result.percentile_numbers().Get(i))),
        result.latency_breakdowns().Get(i));
  }
  return data_table;
}

void GenerateRecommendationResultDataTable(
    const OverviewPage& result, std::unique_ptr<DataTable>& data_table) {
  std::vector<std::vector<std::string>> kColumns = {
      {"tip_type", "string", "tip_type"},
      {"link", "string", "link"},
      {"description", "string", "description"}};

  for (const auto& column : kColumns) {
    data_table->AddColumn(TableColumn(column[0], column[1], column[2]));
  }

  const auto& recommendation = result.recommendation();
  // For each FAQ tip, adds one row to the table.
  for (const auto& faq_tip : recommendation.faq_tips()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("faq");
    row->AddTextCell(faq_tip.link());
    row->AddTextCell("Tool troubleshooting / FAQ");
  }

  // For each host tip, adds one row to the table.
  for (const auto& host_tip : recommendation.host_tips()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("host");
    row->AddTextCell(host_tip.link());
    row->AddTextCell("Next steps for reducing the Host time");
  }
  // For each device tip, adds one row to the table.
  for (const auto& device_tip : recommendation.device_tips()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("device");
    row->AddTextCell(device_tip.link());
    row->AddTextCell("Next steps for reducing the Device time");
  }
  // For each documentation tip, adds one row to the table.
  for (const auto& doc_tip : recommendation.documentation_tips()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("doc");
    row->AddTextCell(doc_tip.link());
    row->AddTextCell("Other useful resources");
  }

  // For each inference tip, adds one row to the table.
  for (const auto& inference_tip : recommendation.inference_tips()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("inference");
    row->AddTextCell(inference_tip.link());
    row->AddTextCell("Recommendations for inference run");
  }

  data_table->AddCustomProperty("bottleneck", recommendation.bottleneck());
  data_table->AddCustomProperty("statement", recommendation.statement());

  GenericRecommendation generic;
  if (recommendation.recommendation().UnpackTo(&generic)) {
    data_table->AddCustomProperty("kernel_launch_bottleneck",
                                  generic.kernel_launch_bottleneck());
    data_table->AddCustomProperty("kernel_launch_statement",
                                  generic.kernel_launch_statement());
    data_table->AddCustomProperty("all_other_bottleneck",
                                  generic.all_other_bottleneck());
    data_table->AddCustomProperty("all_other_statement",
                                  generic.all_other_statement());
    data_table->AddCustomProperty("precision_statement",
                                  generic.precision_statement());
    data_table->AddCustomProperty("device_collectives_bottleneck",
                                  generic.device_collectives_bottleneck());
    data_table->AddCustomProperty("device_collectives_statement",
                                  generic.device_collectives_statement());
  }

  // Prop used to filter out tips on frontend given bottleneck.
  std::vector<std::string> non_bottleneck_tip_types;
  if (recommendation.bottleneck() == "host") {
    non_bottleneck_tip_types.push_back("device");
  } else if (recommendation.bottleneck() == "device") {
    non_bottleneck_tip_types.push_back("host");
  }
  data_table->AddCustomProperty("non_bottleneck_tip_types",
                                absl::StrJoin(non_bottleneck_tip_types, ","));

  // For each input tip, adds one row to the table.
  for (const auto& input_tip : recommendation.input_tips()) {
    TableRow* row = data_table->AddRow();
    row->AddTextCell("input");
    row->AddTextCell(input_tip.link());
    row->AddTextCell("Next steps for reducing the Input time");
  }
}

std::unique_ptr<DataTable> GenerateTpuAnalysisResultToDataTable(
    const OverviewPageAnalysis& analysis) {
  auto analysis_table = std::make_unique<DataTable>();
  AddCommonStats(analysis_table.get(), analysis);

  analysis_table->AddCustomProperty(
      "flop_rate_utilization_relative_to_roofline",
      StrFormatToPercentage(
          analysis.flop_rate_utilization_relative_to_roofline_percent()));

  analysis_table->AddCustomProperty(
      "memory_bw_utilization_relative_to_hw_limit",
      StrFormatToPercentage(
          analysis.memory_bw_utilization_relative_to_hw_limit_percent()));

  analysis_table->AddCustomProperty(
      "device_idle_time_percent",
      StrFormatToPercentage(analysis.device_idle_time_percent()));

  analysis_table->AddCustomProperty(
      "device_duty_cycle_percent",
      StrFormatToPercentage(analysis.device_duty_cycle_percent()));

  analysis_table->AddCustomProperty(
      "host_idle_time_percent",
      StrFormatToPercentage(analysis.host_idle_time_percent()));

  std::vector<std::vector<std::string>> kColumns = {
      {"selfTimePercent", "number", "Time (%)"},
      {"cumulativeTimePercent", "number", "Cumulative time (%)"},
      {"category", "string", "Category"},
      {"operation", "string", "Operation"},
      {"flopRate", "number", "Bf16 Normalized Flop Rate(GFLOPs/Sec)"}};

  for (const auto& column : kColumns) {
    analysis_table->AddColumn(TableColumn(column[0], column[1], column[2]));
  }

  for (const auto& op : analysis.top_device_ops()) {
    TableRow* row = analysis_table->AddRow();
    row->AddNumberCell(op.self_time_fraction());
    row->AddNumberCell(op.cumulative_time_fraction());
    row->AddTextCell(op.category());
    row->AddTextCell(op.name());
    row->AddNumberCell(op.flop_rate());
  }

  return analysis_table;
}

std::unique_ptr<DataTable> GenerateGenericAnalysisResultToDataTable(
    const OverviewPageAnalysis& analysis) {
  auto data_table = std::make_unique<DataTable>();
  AddCommonStats(data_table.get(), analysis);

  std::vector<std::vector<std::string>> kColumns = {
      {"selfTimePercent", "number", "Time (%)"},
      {"cumulativeTimePercent", "number", "Cumulative time (%)"},
      {"category", "string", "Category"},
      {"operation", "string", "Operation"},
      {"flopRate", "number", "GFLOPs/Sec"},
      {"tcEligibility", "boolean", "TensorCore eligibility"},
      {"tcUtilization", "boolean", "Op is using TensorCore"}};

  for (const auto& column : kColumns) {
    data_table->AddColumn(TableColumn(column[0], column[1], column[2]));
  }

  for (const auto& op : analysis.top_device_ops()) {
    TableRow* row = data_table->AddRow();
    row->AddNumberCell(op.self_time_fraction());
    row->AddNumberCell(op.cumulative_time_fraction());
    row->AddTextCell(op.category());
    row->AddTextCell(op.name());
    row->AddNumberCell(op.flop_rate());
    row->AddBooleanCell(op.is_op_tensorcore_eligible());
    row->AddBooleanCell(op.is_op_using_tensorcore());
  }

  data_table->AddCustomProperty(
      "device_compute_16bit_percent",
      StrFormatToPercentage(analysis.device_compute_16bit_percent()));
  data_table->AddCustomProperty(
      "device_compute_32bit_percent",
      StrFormatToPercentage(analysis.device_compute_32bit_percent()));

  return data_table;
}

std::unique_ptr<DataTable> GenerateAnalysisResultDataTable(
    const OverviewPage& result) {
  std::string hardware_type = result.run_environment().device_type();
  if (absl::StrContains(hardware_type, "GPU")) {
    return GenerateGenericAnalysisResultToDataTable(result.analysis());
  } else {
    return GenerateTpuAnalysisResultToDataTable(result.analysis());
  }
}

std::string OverviewPageToJson(const OverviewPage& result) {
  auto analysis_data_table = GenerateAnalysisResultDataTable(result);
  auto steptime_data_table =
      GenerateInputPipelineAnalysisDataTable(result.input_analysis());
  auto run_environment_data_table = GenerateRunEnvironmentDataTable(result);
  auto recommendation_data_table = std::make_unique<DataTable>();
  GenerateRecommendationResultDataTable(result, recommendation_data_table);
  auto inference_latency_data_table =
      GenerateInferenceLatencyDataTable(result.inference_latency());
  auto diagnostics_data_table =
      GenerateDiagnosticsDataTable(result.diagnostics());
  return absl::StrCat("[", analysis_data_table->ToJson(), ",",
                      steptime_data_table.ToJson(), ",",
                      run_environment_data_table->ToJson(), ",",
                      recommendation_data_table->ToJson(), ",",
                      inference_latency_data_table->ToJson(), ", {},",
                      diagnostics_data_table->ToJson(), "]");
}

}  // namespace profiler
}  // namespace tensorflow
