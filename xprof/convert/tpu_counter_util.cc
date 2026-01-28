#include "xprof/convert/tpu_counter_util.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xprof {

TpuCounterUtil::TpuCounterUtil(int host_id, int device_id, int correlation_id,
                               absl::flat_hash_map<uint64_t, uint64_t> counters)
    : host_id_(host_id),
      device_id_(device_id),
      correlation_id_(correlation_id) {
  for (const auto& [id, value] : counters) {
    counters_[id] = {.value = value};
  }
}

const TpuCounter& TpuCounterUtil::GetCounter(uint64_t counter_id) const {
  if (counter_id_ != counter_id) {
    auto it = counters_.find(counter_id);
    if (it == counters_.end()) {
      counter_ = &default_counter_;
    } else {
      counter_ = &it->second;
    }
    counter_id_ = counter_id;
  }
  return *counter_;
}

uint64_t TpuCounterUtil::GetValue(uint64_t counter_id) const {
  return GetCounter(counter_id).value;
}

std::string TpuCounterUtil::DebugString() const {
  std::string result = absl::StrCat("TpuCounterUtil{\n", "host_id_: ", host_id_,
                                    ",\n device_id_: ", device_id_,
                                    ",\n correlation_id_: ", correlation_id_,
                                    ",\n  counters: {\n");

  std::vector<std::string> counter_strings;
  for (const auto& counter : counters_) {
    counter_strings.push_back(
        absl::StrCat("    ", counter.first, ": ", counter.second.value));
  }

  if (!counter_strings.empty()) {
    absl::StrAppend(&result, absl::StrJoin(counter_strings, ",\n"), "\n  }\n}");
  } else {
    absl::StrAppend(&result, "  }\n}");
  }

  return result;
}

std::string UtilizationMetrics::DebugString() const {
  return absl::StrCat("node_id: ", node_id, ", metric: ", metric,
                      ", achieved: ", achieved, ", peak: ", peak,
                      ", unit: ", unit);
}

std::string UtilizationCounters::DebugString() const {
  std::string result = absl::StrCat(
      "UtilizationCounters{\n", "cs_cycles: ", cs_cycles,
      ", num_mxu_inst_issued: ", num_mxu_inst_issued,
      ", num_mxu_busy_cycles: ", num_mxu_busy_cycles,
      ", num_vpu_inst_issued: ", num_vpu_inst_issued,
      ", num_hbm2vmem_bytes: ", num_hbm2vmem_bytes, ", host_id: ", host_id,
      ", device_id: ", device_id, ", correlation_id: ", correlation_id, ",\n");
  for (const auto& metric : metrics) {
    absl::StrAppend(&result, "  ", metric.DebugString(), "\n");
  }
  absl::StrAppend(&result, "}");
  return result;
}

std::ostream& operator<<(std::ostream& os, const TpuCounterUtil& counters) {
  return os << counters.DebugString();
}

std::ostream& operator<<(std::ostream& os,
                         const UtilizationCounters& counters) {
  return os << counters.DebugString();
}

}  // namespace xprof
