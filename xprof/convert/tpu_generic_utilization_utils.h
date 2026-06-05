#ifndef THIRD_PARTY_XPROF_CONVERT_TPU_GENERIC_UTILIZATION_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_TPU_GENERIC_UTILIZATION_UTILS_H_

#include "xprof/convert/tpu_counter_util.h"

namespace xprof {

struct TpuGenericUtilizationOptions {
  int num_mxu_per_tensor_core;
  int cycles_per_xlu_instruction;
  bool is_tpu6e;
  double frequency_hz;
  double peak_hbm_bw_bps;
  double pstate_normalized_frequency_hz;
};

void ComputeTpuGenericTcUnitUtilization(
    const TpuCounterUtil& counters, const TpuGenericUtilizationOptions& options,
    int core, UtilizationCounters* utilization);

void ComputeTpuGenericBandwidthUtilization(
    const TpuCounterUtil& counters, const TpuGenericUtilizationOptions& options,
    int core, UtilizationCounters* utilization);

void ComputeTpuGenericIciBandwidthUtilization(
    const TpuCounterUtil& counters, const TpuGenericUtilizationOptions& options,
    UtilizationCounters* utilization);

void ComputeTpuv7xScUnitUtilization(const TpuCounterUtil& counters, int die,
                                    int core, UtilizationCounters* utilization);

void ComputeTpuv6eScUnitUtilization(const TpuCounterUtil& counters, int die,
                                    int core, UtilizationCounters* utilization);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_TPU_GENERIC_UTILIZATION_UTILS_H_
