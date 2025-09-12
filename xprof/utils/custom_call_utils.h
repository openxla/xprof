#ifndef THIRD_PARTY_XPROF_UTILS_CUSTOM_CALL_UTILS_H_
#define THIRD_PARTY_XPROF_UTILS_CUSTOM_CALL_UTILS_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xprof {

// Returns the custom call text from the HloInstruction for XLA:TPU.
absl::StatusOr<std::string> GetCustomCallText(
    const xla::HloInstruction& hlo_instruction);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_UTILS_CUSTOM_CALL_UTILS_H_
