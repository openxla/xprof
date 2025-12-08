#include "xprof/utils/tpu_counter_util.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace xprof {

TpuCounterUtil::TpuCounterUtil(
    int host_id, int device_id, int correlation_id,
    const absl::flat_hash_map<uint64_t, uint64_t>& counters)
    : host_id_(host_id),
      device_id_(device_id),
      correlation_id_(correlation_id),
      counters_(counters) {}

const uint64_t TpuCounterUtil::GetCounter(uint64_t counter_id) const {
  auto it = counters_.find(counter_id);
  if (it == counters_.end()) {
    return 0;
  }
  return it->second;
}

uint64_t TpuCounterUtil::GetValue(uint64_t counter_id) const {
  return GetCounter(counter_id);
}

std::string TpuCounterUtil::DebugString() const {
  std::string result = absl::StrCat(
      "TpuCounterUtil{\n", "host_id: ", host_id_, ",\n",
      "device_id: ", device_id_, ",\n", "correlation_id: ", correlation_id_,
      ",\n  counters: {\n");

  std::vector<std::string> counter_strings;
  for (const auto& counter : counters_) {
    counter_strings.push_back(
        absl::StrCat("    ", counter.first, ": ", counter.second));
  }

  if (!counter_strings.empty()) {
    absl::StrAppend(&result, absl::StrJoin(counter_strings, ",\n"), "\n  }\n}");
  } else {
    absl::StrAppend(&result, "  }\n}");
  }

  return result;
}

}  // namespace xprof
