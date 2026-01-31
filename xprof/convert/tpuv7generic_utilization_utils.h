#ifndef THIRD_PARTY_XPROF_CONVERT_TPUV7GENERIC_UTILIZATION_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_TPUV7GENERIC_UTILIZATION_UTILS_H_

#include "xprof/convert/tpu_counter_util.h"

namespace xprof {

struct Tpuv7GenericUtilizationOptions {
  int num_mxu_per_tensor_core;
  int cycles_per_xlu_instruction;
  bool is_tpu7;
};

void ComputeTpuv7GenericTcUnitUtilization(
    const TpuCounterUtil& counters,
    const Tpuv7GenericUtilizationOptions& options, int core,
    UtilizationCounters* utilization);

void ComputeTpuv7xScUnitUtilization(const TpuCounterUtil& counters, int die,
                                    int core, UtilizationCounters* utilization);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_TPUV7GENERIC_UTILIZATION_UTILS_H_
