#ifndef XPROF_CONVERT_TPU_COUNTER_UTIL_H_
#define XPROF_CONVERT_TPU_COUNTER_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace xprof {

// This struct is used to store TPU performance counter values.
struct TpuCounter {
  uint64_t value = 0;
};

class TpuCounterUtil {
 public:
  TpuCounterUtil(int host_id, int device_id, int correlation_id,
                 absl::flat_hash_map<uint64_t, uint64_t> counters);

  int host_id() const { return host_id_; }
  int device_id() const { return device_id_; }
  int correlation_id() const { return correlation_id_; }

  // Returns a counter value.
  uint64_t GetValue(uint64_t counter_id) const;

  std::string DebugString() const;

 private:
  // Finds a counter in counters_ and returns it.
  // If the counter is not present, the default instance is returned.
  const TpuCounter& GetCounter(uint64_t counter_id) const;

  const int host_id_;
  const int device_id_;
  const int correlation_id_;

  // Counters collected.
  absl::flat_hash_map<uint64_t, TpuCounter> counters_;
  const TpuCounter default_counter_{};

  // Most recent result of GetCounter.
  mutable uint64_t counter_id_ = -1;
  mutable const TpuCounter* counter_ = &default_counter_;
};

struct UtilizationMetrics {
  uint64_t node_id = 0;
  std::string metric;
  double achieved = 0.0;
  double peak = 0.0;
  std::string unit;

  std::string DebugString() const;
};

// The numbers are aggregated among cores of the chip that was wrapped by
// TpuCounterUtil.
struct UtilizationCounters {
  // The number of cycles of tensor cores on the chip.
  uint64_t cs_cycles = 0;
  // The number of all MXU instructions issued on the chip.
  uint64_t num_mxu_inst_issued = 0;
  // The number of all MXU busy cycles on the chip.
  uint64_t num_mxu_busy_cycles = 0;
  // The number of VPU instructions issued.
  uint64_t num_vpu_inst_issued = 0;
  // The number of DMA operations (from HBM to VMEM) issued.
  uint64_t num_hbm2vmem_bytes = 0;

  uint64_t host_id = 0;
  uint64_t device_id = 0;
  uint64_t correlation_id = 0;
  std::vector<UtilizationMetrics> metrics;

  std::string DebugString() const;
};

std::ostream& operator<<(std::ostream& os, const TpuCounterUtil& counters);
std::ostream& operator<<(std::ostream& os, const UtilizationCounters& counters);

}  // namespace xprof

#endif  // XPROF_CONVERT_TPU_COUNTER_UTIL_H_
