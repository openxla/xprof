#ifndef THIRD_PARTY_XPROF_CONVERT_CUSTOM_CALL_DUTY_CYCLE_H_
#define THIRD_PARTY_XPROF_CONVERT_CUSTOM_CALL_DUTY_CYCLE_H_

#include <cstdint>

#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace xprof {

// The absolute source of truth for the Duty Cycle CustomCall logic.
// Returns true if the custom call is considered off-duty.
bool IsCustomCallOffDuty(int64_t flops, int64_t model_flops, double flops_v2,
                         double model_flops_v2, bool has_ici);

// Returns true if the event represents an off-duty custom call.
bool IsCustomCallEventOffDuty(const tsl::profiler::XEventVisitor& event);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_CUSTOM_CALL_DUTY_CYCLE_H_
