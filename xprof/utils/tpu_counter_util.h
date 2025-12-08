#ifndef THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_UTIL_H_
#define THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xprof/utils/tpu_counter_ids.h"

namespace xprof {

class TpuCounterUtil {
 public:
  // Populates counters_ from an id->value map.
  TpuCounterUtil(int host_id, int device_id,
                 int correlation_id,
                 const absl::flat_hash_map<uint64_t, uint64_t>& counters);
  int host_id() const { return host_id_; }
  int device_id() const { return device_id_; }
  int correlation_id() const { return correlation_id_; }

  // Returns a counter value.
  uint64_t GetValue(uint64_t counter_id) const;

  std::string DebugString() const;

 private:
  // Finds a counter in counters_ and returns it.
  // If the counter is not present, the default instance is returned.
  const uint64_t GetCounter(
      uint64_t counter_id) const;
  const int host_id_;
  const int device_id_;
  const int correlation_id_;

  // Counters collected in driver proto format.
  absl::flat_hash_map<uint64_t, uint64_t> counters_;
};

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_UTIL_H_
