#include "xprof/convert/custom_call_duty_cycle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace xprof {
namespace {

constexpr double kEpsilon = 1e-9;

// Compares floating-point values using a hybrid (absolute + relative)
// tolerance. Uses an absolute epsilon near zero (where relative comparison
// fails) and scales relatively for larger numbers to account for IEEE 754
// floating-point precision limits.
bool ApproximatelyEqual(double a, double b, double epsilon = kEpsilon) {
  return std::abs(a - b) <= epsilon * std::max({1.0, std::abs(a), std::abs(b)});
}

std::optional<tsl::profiler::XStatVisitor> GetXEventStat(
    const tsl::profiler::XEventVisitor& event,
    tsl::profiler::StatType stat_type) {
  std::optional<tsl::profiler::XStatVisitor> stat =
      event.Metadata().GetStat(stat_type);
  if (!stat.has_value()) {
    stat = event.GetStat(stat_type);
  }
  return stat;
}

int64_t GetIntStatDef(const tsl::profiler::XEventVisitor& event,
                      tsl::profiler::StatType stat_type, int64_t def) {
  auto stat = GetXEventStat(event, stat_type);
  return stat.has_value() ? stat->IntOrUintValue() : def;
}

}  // namespace

bool IsCustomCallOffDuty(int64_t flops, int64_t model_flops, double flops_v2,
                         double model_flops_v2, bool has_ici) {
  return (flops == 0) && (model_flops == 0) &&
         ApproximatelyEqual(flops_v2, 0.0) &&
         ApproximatelyEqual(model_flops_v2, 0.0) && !has_ici;
}

bool IsCustomCallEventOffDuty(const tsl::profiler::XEventVisitor& event) {
  std::optional<tsl::profiler::XStatVisitor> category =
      GetXEventStat(event, tsl::profiler::StatType::kHloCategory);
  if (!category.has_value() ||
      category->StrOrRefValue() !=
          xla::HloOpcodeString(xla::HloOpcode::kCustomCall)) {
    return false;
  }

  bool has_ici = GetIntStatDef(event, tsl::profiler::StatType::kUsesIci, 0) > 0;
  int64_t flops = GetIntStatDef(event, tsl::profiler::StatType::kFlops, 0);
  int64_t model_flops =
      GetIntStatDef(event, tsl::profiler::StatType::kModelFlops, 0);
  return IsCustomCallOffDuty(flops, model_flops, /*flops_v2=*/0.0,
                             /*model_flops_v2=*/0.0, has_ici);
}

}  // namespace xprof
