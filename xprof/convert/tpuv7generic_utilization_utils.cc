#include "xprof/convert/tpuv7generic_utilization_utils.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/tpu_counter_util.h"
#include "xprof/utils/tpu_counter_ids_v7.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {

namespace {

using Tpu7CounterName = TpuCounterIdsTpu7;
using Tpuv7xCounterName = TpuCounterIdsTpu7x;

// String constants for units.
constexpr absl::string_view kInstructions = "instructions";
constexpr absl::string_view kCycles = "cycles";
constexpr absl::string_view kBytes = "bytes";
constexpr absl::string_view kPercent = "percent";

void AddUtilization(const TpuCounterUtil& counters, uint64_t node_id,
                    absl::string_view metric, double achieved, double peak,
                    absl::string_view unit, UtilizationCounters* utilization) {
  utilization->host_id = counters.host_id();
  utilization->device_id = counters.device_id();
  utilization->correlation_id = counters.correlation_id();
  utilization->metrics.push_back(UtilizationMetrics{
      node_id, std::string(metric), achieved, peak, std::string(unit)});
}

}  // namespace

void ComputeTpuv7GenericTcUnitUtilization(
    const TpuCounterUtil& counters,
    const Tpuv7GenericUtilizationOptions& options, int core,
    UtilizationCounters* utilization) {
  bool is_tpu7 = options.is_tpu7;
  // NOLINTBEGIN
#define TPU7_COUNTER(NAME) \
  TpuCounterIdsTpu7::      \
      VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME
#define TPUV7X_COUNTER(NAME)                                                                    \
  ((core == 0)                                                                                  \
       ? TpuCounterIdsTpu7x::                                                                   \
             VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME \
       : TpuCounterIdsTpu7x::                                                                   \
             VF_CHIP_DIE1_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME)
#define TPUV7X_PWRMGR_COUNTER_ADDR(NAME)                                                 \
  ((core == 0)                                                                           \
       ? TpuCounterIdsTpu7x::                                                            \
             VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_##NAME \
       : TpuCounterIdsTpu7x::                                                            \
             VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_##NAME)

#define COUNTER(NAME) \
  counters.GetValue(is_tpu7 ? (TPU7_COUNTER(NAME)) : (TPUV7X_COUNTER(NAME)))

  // NOLINTEND
  // For TPUv7x, we can have throttled cycles in the tensor core.
  uint64_t cycles =
      is_tpu7 ? COUNTER(CYCLES)
              : counters.GetValue(TPUV7X_PWRMGR_COUNTER_ADDR(CYCLE_COUNT));
  if (cycles == 0) return;

  uint64_t clocks_skipped =
      counters.GetValue(TPUV7X_PWRMGR_COUNTER_ADDR(CLOCKS_SKIPPED));
  uint64_t ext_throttle_clocks_skipped = counters.GetValue(
      TPUV7X_PWRMGR_COUNTER_ADDR(EXT_THROTTLE_CLOCKS_SKIPPED));
  uint64_t ldidt_droop_clocks_skipped =
      counters.GetValue(TPUV7X_PWRMGR_COUNTER_ADDR(LDIDT_DROOP_CLOCKS_SKIPPED));

  if (!is_tpu7) {
    AddUtilization(counters, core, "Clocks Skipped", clocks_skipped, cycles,
                   kCycles, utilization);
    AddUtilization(counters, core, "Ext Throttle Clocks Skipped",
                   ext_throttle_clocks_skipped, cycles, kCycles, utilization);
    AddUtilization(counters, core, "LDIDT Droop Clocks Skipped",
                   ldidt_droop_clocks_skipped, cycles, kCycles, utilization);
  }

  utilization->cs_cycles += cycles;

  // Scalar Unit.
  // This assumes that all scalar instructions take one cycle, which may be
  // incorrect.
  AddUtilization(
      counters, core, "Scalar Unit",
      COUNTER(SCALAR_ALU_INSTRUCTION_0) + COUNTER(SCALAR_ALU_INSTRUCTION_1),
      cycles * 2, kInstructions, utilization);

  // Vector Unit.
  // This assumes that all vector instructions take one cycle, which may be
  // incorrect.
  const uint64_t kVectorALUsPerCore = 4;
  uint64_t num_vpu_inst_issued =
      COUNTER(VECTOR_ALU_INSTRUCTION_0) + COUNTER(VECTOR_ALU_INSTRUCTION_1) +
      COUNTER(VECTOR_ALU_INSTRUCTION_2) + COUNTER(VECTOR_ALU_INSTRUCTION_3);
  utilization->num_vpu_inst_issued += num_vpu_inst_issued;
  AddUtilization(counters, core, "Vector ALUs", num_vpu_inst_issued,
                 cycles * kVectorALUsPerCore, kInstructions, utilization);

  // Loads and stores take one cycle.
  // TODO(b/271325351): Should we consider VMISC_VST_INSTRUCTION as well?
  AddUtilization(counters, core, "Vmem Stores", COUNTER(VST_INSTRUCTION),
                 cycles, kInstructions, utilization);
  // TODO(b/271325351): Should we consider VMISC_VLD_INSTRUCTION as well?
  AddUtilization(counters, core, "Vmem Loads",
                 COUNTER(VLD_INSTRUCTION_0) + COUNTER(VLD_INSTRUCTION_1),
                 cycles * 2, kInstructions, utilization);

  // Multiply Unit.
  AddUtilization(counters, core, "No MXU Busy", COUNTER(MXU_BUSY_0), cycles,
                 kCycles, utilization);
  AddUtilization(counters, core, "1 MXU Busy", COUNTER(MXU_BUSY_1), cycles,
                 kCycles, utilization);
  AddUtilization(counters, core, "2 MXU Busy", COUNTER(MXU_BUSY_2), cycles,
                 kCycles, utilization);
  AddUtilization(counters, core, "Avg MXU Busy",
                 0.5 * COUNTER(MXU_BUSY_1) + COUNTER(MXU_BUSY_2), cycles,
                 kCycles, utilization);

  // CPI info can be found at
#define GET_TPU7_COUNTER(NAME) counters.GetValue(TPU7_COUNTER(NAME))
#define COMPUTE_TPU7_MXU_BF16_CYCLES(unit_id)                             \
  kCpiMxuLmrBf16* GET_TPU7_COUNTER(MATMUL_LMR_BF16_MXU_##unit_id) +       \
      kCpiMxuVregF8* GET_TPU7_COUNTER(MATMUL_VREG_F8_MXU_##unit_id) +     \
      kCpiMxuVregBf16* GET_TPU7_COUNTER(MATMUL_VREG_BF16_MXU_##unit_id) + \
      kCpiMxuVregF32* GET_TPU7_COUNTER(MATMUL_VREG_F32_MXU_##unit_id)
#define COMPUTE_TPU7_MXU_I8_CYCLES(unit_id)                     \
  kCpiMxuLmrI8* GET_TPU7_COUNTER(MATMUL_LMR_I8_MXU_##unit_id) + \
      kCpiMxuVregI8* GET_TPU7_COUNTER(MATMUL_VREG_I8_MXU_##unit_id)
#define COMPUTE_TPU7_MXU_I4_CYCLES(unit_id)                     \
  kCpiMxuLmrI4* GET_TPU7_COUNTER(MATMUL_LMR_I4_MXU_##unit_id) + \
      kCpiMxuVregI4* GET_TPU7_COUNTER(MATMUL_VREG_I4_MXU_##unit_id)
#define COMPUTE_TPU7_MXU_CYCLES(unit_id)    \
  COMPUTE_TPU7_MXU_BF16_CYCLES(unit_id) +   \
      COMPUTE_TPU7_MXU_I8_CYCLES(unit_id) + \
      COMPUTE_TPU7_MXU_I4_CYCLES(unit_id)
#define COMPUTE_TPU7_MXU_INSTRUCTIONS(unit_id)        \
  (GET_TPU7_COUNTER(MATMUL_LMR_BF16_MXU_##unit_id) +  \
   GET_TPU7_COUNTER(MATMUL_LMR_I8_MXU_##unit_id) +    \
   GET_TPU7_COUNTER(MATMUL_LMR_I4_MXU_##unit_id) +    \
   GET_TPU7_COUNTER(MATMUL_VREG_I4_MXU_##unit_id) +   \
   GET_TPU7_COUNTER(MATMUL_VREG_I8_MXU_##unit_id) +   \
   GET_TPU7_COUNTER(MATMUL_VREG_F8_MXU_##unit_id) +   \
   GET_TPU7_COUNTER(MATMUL_VREG_BF16_MXU_##unit_id) + \
   GET_TPU7_COUNTER(MATMUL_VREG_F32_MXU_##unit_id))

    utilization->num_mxu_inst_issued +=
        COMPUTE_TPU7_MXU_INSTRUCTIONS(0) + COMPUTE_TPU7_MXU_INSTRUCTIONS(1);
    utilization->num_mxu_busy_cycles +=
        COMPUTE_TPU7_MXU_CYCLES(0) + COMPUTE_TPU7_MXU_CYCLES(1);

    AddUtilization(counters, core, "MXU0", COMPUTE_TPU7_MXU_CYCLES(0), cycles,
                   kCycles, utilization);
    AddUtilization(counters, core, "MXU1", COMPUTE_TPU7_MXU_CYCLES(1), cycles,
                   kCycles, utilization);

    AddUtilization(
        counters, core, "MXU BF16",
        COMPUTE_TPU7_MXU_BF16_CYCLES(0) + COMPUTE_TPU7_MXU_BF16_CYCLES(1),
        cycles * kNumMxuPerTC, kCycles, utilization);

    AddUtilization(
        counters, core, "MXU I8",
        COMPUTE_TPU7_MXU_I8_CYCLES(0) + COMPUTE_TPU7_MXU_I8_CYCLES(1),
        cycles * kNumMxuPerTC, kCycles, utilization);
    AddUtilization(
        counters, core, "MXU I4",
        COMPUTE_TPU7_MXU_I4_CYCLES(0) + COMPUTE_TPU7_MXU_I4_CYCLES(1),
        cycles * kNumMxuPerTC, kCycles, utilization);

#undef GET_TPU7_COUNTER
#undef COMPUTE_TPU7_MXU_INSTRUCTIONS
#undef COMPUTE_TPU7_MXU_CYCLES
#undef COMPUTE_TPU7_MXU_BF16_CYCLES
#undef COMPUTE_TPU7_MXU_I8_CYCLES
#undef COMPUTE_TPU7_MXU_I4_CYCLES

  } else {
#define GET_TPUV7X_COUNTER(NAME) counters.GetValue(TPUV7X_COUNTER(NAME))
#define COMPUTE_TPUV7X_MXU_BF16_CYCLES(unit_id)                             \
  kCpiMxuLmrBf16* GET_TPUV7X_COUNTER(MATMUL_LMR_BF16_MXU_##unit_id) +       \
      kCpiMxuVregBf16* GET_TPUV7X_COUNTER(MATMUL_VREG_BF16_MXU_##unit_id) + \
      kCpiMxuVregF32* GET_TPUV7X_COUNTER(MATMUL_VREG_F32_MXU_##unit_id)
#define COMPUTE_TPUV7X_MXU_E5M2_E4M3_CYCLES(unit_id) \
  kCpiMxuVregF8* GET_TPUV7X_COUNTER(MATMUL_VREG_F8_MXU_##unit_id)
#define COMPUTE_TPUV7X_MXU_CYCLES(unit_id)  \
  COMPUTE_TPUV7X_MXU_BF16_CYCLES(unit_id) + \
      COMPUTE_TPUV7X_MXU_E5M2_E4M3_CYCLES(unit_id)
#define COMPUTE_TPUV7X_MXU_INSTRUCTIONS(unit_id)        \
  (GET_TPUV7X_COUNTER(MATMUL_LMR_BF16_MXU_##unit_id) +  \
   GET_TPUV7X_COUNTER(MATMUL_VREG_F8_MXU_##unit_id) +   \
   GET_TPUV7X_COUNTER(MATMUL_VREG_BF16_MXU_##unit_id) + \
   GET_TPUV7X_COUNTER(MATMUL_VREG_F32_MXU_##unit_id))

    utilization->num_mxu_inst_issued +=
        COMPUTE_TPUV7X_MXU_INSTRUCTIONS(0) + COMPUTE_TPUV7X_MXU_INSTRUCTIONS(1);
    utilization->num_mxu_busy_cycles +=
        COMPUTE_TPUV7X_MXU_CYCLES(0) + COMPUTE_TPUV7X_MXU_CYCLES(1);

    AddUtilization(counters, core, "MXU0", COMPUTE_TPUV7X_MXU_CYCLES(0), cycles,
                   kCycles, utilization);
    AddUtilization(counters, core, "MXU1", COMPUTE_TPUV7X_MXU_CYCLES(1), cycles,
                   kCycles, utilization);

    AddUtilization(
        counters, core, "MXU BF16",
        COMPUTE_TPUV7X_MXU_BF16_CYCLES(0) + COMPUTE_TPUV7X_MXU_BF16_CYCLES(1),
        cycles * kNumMxuPerTC, kCycles, utilization);

    AddUtilization(counters, core, "MXU E4M3 + E5M2",
                   COMPUTE_TPUV7X_MXU_E5M2_E4M3_CYCLES(0) +
                       COMPUTE_TPUV7X_MXU_E5M2_E4M3_CYCLES(1),
                   cycles * kNumMxuPerTC, kCycles, utilization);

#undef GET_TPUV7X_COUNTER
#undef COMPUTE_TPUV7X_MXU_INSTRUCTIONS
#undef COMPUTE_TPUV7X_MXU_CYCLES
#undef COMPUTE_TPUV7X_MXU_BF16_CYCLES
  }

  // MXU Matpush
  const uint64_t matpush_cycles =
      COUNTER(MATPUSH_CYCLES_MXU_0) + COUNTER(MATPUSH_CYCLES_MXU_1);
  if (cycles > 0) {
    AddUtilization(counters, core, "MXU matpush", matpush_cycles,
                   cycles * kNumMxuPerTC, kCycles, utilization);
  }

  // Cross Lane Unit. (XLU)
  AddUtilization(counters, core, "No XLU Busy", COUNTER(XLU_BUSY_0), cycles,
                 kCycles, utilization);
  AddUtilization(counters, core, "1 XLU Busy", COUNTER(XLU_BUSY_1), cycles,
                 kCycles, utilization);
  AddUtilization(counters, core, "2 XLUs Busy", COUNTER(XLU_BUSY_2), cycles,
                 kCycles, utilization);
  AddUtilization(counters, core, "Avg XLU Busy",
                 0.5 * COUNTER(XLU_BUSY_1) + COUNTER(XLU_BUSY_2), cycles,
                 kCycles, utilization);
  const int cycles_per_xlu_instruction = options.cycles_per_xlu_instruction;
  // TODO(b/271325351): Double check if "PACKED_XLU_N" should be included here.
  AddUtilization(
      counters, core, "XLU0",
      COUNTER(PACKED_XLU_0) + COUNTER(ROTATE_PERMUTE_INSTRUCTION_XLU_0) +
          COUNTER(ROTATE_PERMUTE_SET_INSTRUCTION_XLU_0) +
          COUNTER(TRANSPOSE_XLU_0),
      cycles / cycles_per_xlu_instruction, kInstructions, utilization);
  AddUtilization(
      counters, core, "XLU1",
      COUNTER(PACKED_XLU_1) + COUNTER(ROTATE_PERMUTE_INSTRUCTION_XLU_1) +
          COUNTER(ROTATE_PERMUTE_SET_INSTRUCTION_XLU_1) +
          COUNTER(TRANSPOSE_XLU_1),
      cycles / cycles_per_xlu_instruction, kInstructions, utilization);

#undef COUNTER
}

// NOLINTBEGIN
#define TPUV7X_SCS_COUNTER(DIE, CORE, NAME)                                         \
  DIE == 0                                                                          \
      ? CORE == 0                                                                   \
            ? Tpuv7xCounterName::                                                   \
                  VF_CHIP_DIE0_SC_0_SCS_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME \
            : Tpuv7xCounterName::                                                   \
                  VF_CHIP_DIE0_SC_1_SCS_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME \
  : CORE == 0                                                                       \
      ? Tpuv7xCounterName::                                                         \
            VF_CHIP_DIE1_SC_0_SCS_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME       \
      : Tpuv7xCounterName::                                                         \
            VF_CHIP_DIE1_SC_1_SCS_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME
#define TPUV7X_SCT_COUNTER(DIE, CORE, TILE, SUBSYSTEM, NAME)                                           \
  DIE == 0                                                                                             \
      ? CORE == 0                                                                                      \
            ? Tpuv7xCounterName::                                                                      \
                  VF_CHIP_DIE0_SC_0_##SUBSYSTEM##_##TILE##_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME \
            : Tpuv7xCounterName::                                                                      \
                  VF_CHIP_DIE0_SC_1_##SUBSYSTEM##_##TILE##_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME \
  : CORE == 0                                                                                          \
      ? Tpuv7xCounterName::                                                                            \
            VF_CHIP_DIE1_SC_0_##SUBSYSTEM##_##TILE##_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME       \
      : Tpuv7xCounterName::                                                                            \
            VF_CHIP_DIE1_SC_1_##SUBSYSTEM##_##TILE##_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_##NAME
// NOLINTEND
#define DECLARE_16_TILES(DIE, CORE, SUBSYSTEM, NAME)      \
  TPUV7X_SCT_COUNTER(DIE, CORE, 0, SUBSYSTEM, NAME),      \
      TPUV7X_SCT_COUNTER(DIE, CORE, 1, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 2, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 3, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 4, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 5, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 6, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 7, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 8, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 9, SUBSYSTEM, NAME),  \
      TPUV7X_SCT_COUNTER(DIE, CORE, 10, SUBSYSTEM, NAME), \
      TPUV7X_SCT_COUNTER(DIE, CORE, 11, SUBSYSTEM, NAME), \
      TPUV7X_SCT_COUNTER(DIE, CORE, 12, SUBSYSTEM, NAME), \
      TPUV7X_SCT_COUNTER(DIE, CORE, 13, SUBSYSTEM, NAME), \
      TPUV7X_SCT_COUNTER(DIE, CORE, 14, SUBSYSTEM, NAME), \
      TPUV7X_SCT_COUNTER(DIE, CORE, 15, SUBSYSTEM, NAME)

void ComputeTpuv7xScUnitUtilization(const TpuCounterUtil& counters, int die,
                                    int core,
                                    UtilizationCounters* utilization) {
  const std::array<uint64_t, 16> tpuv7x_sc_tec_scalar_issue = {
      DECLARE_16_TILES(die, core, SCTD, TEC_SCALAR_ISSUE)};

  // A finer-grain stats can be provided for scalar issue count
  const std::array<uint64_t, 16> tpuv7x_sc_tec_s0_instruction = {
      DECLARE_16_TILES(die, core, SCTD, TEC_S0_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_tec_s1_instruction = {
      DECLARE_16_TILES(die, core, SCTD, TEC_S1_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_tec_smisc_instruction = {
      DECLARE_16_TILES(die, core, SCTD, TEC_SMISC_INSTRUCTION)};

  const uint64_t tec_cycle =
      counters.GetValue(TPUV7X_SCT_COUNTER(die, core, 0, SCTD, CYCLES));

  const uint64_t sc_tec_scalar_issue_count =
      absl::c_accumulate(tpuv7x_sc_tec_scalar_issue, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "TEC Scalar Issue",
                 sc_tec_scalar_issue_count, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_tec_s0_issue_count =
      absl::c_accumulate(tpuv7x_sc_tec_s0_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "TEC Scalar Issue: S0 slot",
                 sc_tec_s0_issue_count, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_tec_s1_issue_count =
      absl::c_accumulate(tpuv7x_sc_tec_s1_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "TEC Scalar Issue: S1 slot",
                 sc_tec_s1_issue_count, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_tec_smisc_issue_count =
      absl::c_accumulate(tpuv7x_sc_tec_smisc_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "TEC Scalar Issue: Smisc slot",
                 sc_tec_smisc_issue_count, tec_cycle * 16, kInstructions,
                 utilization);

  // Each sparsecore has 16 tiles where each tile has a dedicated performance
  // counter to count vector instruction issues.
  const std::array<uint64_t, 16> tpuv7x_sc_vector_instruction = {
      DECLARE_16_TILES(die, core, SCTC, VECTOR_ISSUE)};
  // A finer-grain stats can be provided for vector instruction count
  const std::array<uint64_t, 16> tpuv7x_sc_vector_v0_instruction = {
      DECLARE_16_TILES(die, core, SCTC, V0_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_vector_v1_instruction = {
      DECLARE_16_TILES(die, core, SCTC, V1_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_vector_v2_instruction = {
      DECLARE_16_TILES(die, core, SCTC, V2_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_vector_vld_instruction = {
      DECLARE_16_TILES(die, core, SCTC, VLD_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_vector_vst_instruction = {
      DECLARE_16_TILES(die, core, SCTC, VST_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_vector_vres_instruction = {
      DECLARE_16_TILES(die, core, SCTC, VRES_INSTRUCTION)};

  const std::array<uint64_t, 16> tpuv7x_sc_vector_vex_instruction = {
      DECLARE_16_TILES(die, core, SCTC, VEX_INSTRUCTION)};

  const uint64_t sc_vector_issue_count =
      absl::c_accumulate(tpuv7x_sc_vector_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue",
                 sc_vector_issue_count, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_v0_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_v0_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: V0 Slot",
                 sc_vector_v0_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_v1_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_v1_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: V1 Slot",
                 sc_vector_v1_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_v2_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_v2_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: V2 Slot",
                 sc_vector_v2_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_vld_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_vld_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: VLD Slot",
                 sc_vector_vld_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_vst_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_vst_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: VST Slot",
                 sc_vector_vst_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_vex_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_vex_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: VEX Slot",
                 sc_vector_vex_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  const uint64_t sc_vector_vres_instruction =
      absl::c_accumulate(tpuv7x_sc_vector_vres_instruction, 0ull,
                         [&](uint64_t sum, const uint64_t counter_id) {
                           return sum + counters.GetValue(counter_id);
                         });

  AddUtilization(counters, (die << 1) + core, "SC Vector Issue: VRES Slot",
                 sc_vector_vres_instruction, tec_cycle * 16, kInstructions,
                 utilization);

  // SCS Counters
  const uint64_t scs_cycles =
      counters.GetValue(TPUV7X_SCS_COUNTER(die, core, CYCLES));
  AddUtilization(counters, (die << 1) + core, "SCS Scalar Issue",
                 counters.GetValue(TPUV7X_SCS_COUNTER(die, core, SCALAR_ISSUE)),
                 scs_cycles, kInstructions, utilization);

  AddUtilization(
      counters, (die << 1) + core, "SCS: S0 Slot",
      counters.GetValue(TPUV7X_SCS_COUNTER(die, core, S0_INSTRUCTION)),
      scs_cycles, kInstructions, utilization);

  AddUtilization(
      counters, (die << 1) + core, "SCS: S1 Slot",
      counters.GetValue(TPUV7X_SCS_COUNTER(die, core, S1_INSTRUCTION)),
      scs_cycles, kInstructions, utilization);

  AddUtilization(
      counters, (die << 1) + core, "SCS: Smisc Slot",
      counters.GetValue(TPUV7X_SCS_COUNTER(die, core, SMISC_INSTRUCTION)),
      scs_cycles, kInstructions, utilization);
}
#undef TPUV7X_SCS_COUNTER
#undef TPUV7X_SCT_COUNTER
#undef DECLARE_16_TILES

// Helper macros for HBM counters (TPU v7x)
#define HBM_PREFIX(DIE, id) TpuCounterIdsTpu7x::VF_CHIP_DIE##DIE##_##id
#define RD_RESP_PS0(DIE, id) \
  HBM_PREFIX(DIE, id##_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_RD_RESP_PS0)
#define RD_RESP_PS1(DIE, id) \
  HBM_PREFIX(DIE, id##_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_RD_RESP_PS1)
#define WR_REQ_PS0(DIE, id)                                                 \
  HBM_PREFIX(DIE, id##_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_WR_REQ_PS0), \
      HBM_PREFIX(                                                           \
          DIE,                                                              \
          id##_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_PARTIAL_WRITE_REQ_PS0)
#define WR_REQ_PS1(DIE, id)                                                 \
  HBM_PREFIX(DIE, id##_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_WR_REQ_PS1), \
      HBM_PREFIX(                                                           \
          DIE,                                                              \
          id##_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_PARTIAL_WRITE_REQ_PS1)

std::vector<uint64_t> GetTpuv7xHbmReadCounters(int die) {
  if (die == 0) {
    return {
        RD_RESP_PS0(0, HBM_0_SS_HBMC_0),  RD_RESP_PS1(0, HBM_0_SS_HBMC_0),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_1),  RD_RESP_PS1(0, HBM_0_SS_HBMC_1),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_2),  RD_RESP_PS1(0, HBM_0_SS_HBMC_2),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_3),  RD_RESP_PS1(0, HBM_0_SS_HBMC_3),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_4),  RD_RESP_PS1(0, HBM_0_SS_HBMC_4),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_5),  RD_RESP_PS1(0, HBM_0_SS_HBMC_5),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_6),  RD_RESP_PS1(0, HBM_0_SS_HBMC_6),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_7),  RD_RESP_PS1(0, HBM_0_SS_HBMC_7),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_8),  RD_RESP_PS1(0, HBM_0_SS_HBMC_8),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_9),  RD_RESP_PS1(0, HBM_0_SS_HBMC_9),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_10), RD_RESP_PS1(0, HBM_0_SS_HBMC_10),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_11), RD_RESP_PS1(0, HBM_0_SS_HBMC_11),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_12), RD_RESP_PS1(0, HBM_0_SS_HBMC_12),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_13), RD_RESP_PS1(0, HBM_0_SS_HBMC_13),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_14), RD_RESP_PS1(0, HBM_0_SS_HBMC_14),
        RD_RESP_PS0(0, HBM_0_SS_HBMC_15), RD_RESP_PS1(0, HBM_0_SS_HBMC_15),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_0),  RD_RESP_PS1(0, HBM_1_SS_HBMC_0),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_1),  RD_RESP_PS1(0, HBM_1_SS_HBMC_1),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_2),  RD_RESP_PS1(0, HBM_1_SS_HBMC_2),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_3),  RD_RESP_PS1(0, HBM_1_SS_HBMC_3),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_4),  RD_RESP_PS1(0, HBM_1_SS_HBMC_4),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_5),  RD_RESP_PS1(0, HBM_1_SS_HBMC_5),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_6),  RD_RESP_PS1(0, HBM_1_SS_HBMC_6),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_7),  RD_RESP_PS1(0, HBM_1_SS_HBMC_7),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_8),  RD_RESP_PS1(0, HBM_1_SS_HBMC_8),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_9),  RD_RESP_PS1(0, HBM_1_SS_HBMC_9),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_10), RD_RESP_PS1(0, HBM_1_SS_HBMC_10),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_11), RD_RESP_PS1(0, HBM_1_SS_HBMC_11),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_12), RD_RESP_PS1(0, HBM_1_SS_HBMC_12),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_13), RD_RESP_PS1(0, HBM_1_SS_HBMC_13),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_14), RD_RESP_PS1(0, HBM_1_SS_HBMC_14),
        RD_RESP_PS0(0, HBM_1_SS_HBMC_15), RD_RESP_PS1(0, HBM_1_SS_HBMC_15),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_0),  RD_RESP_PS1(0, HBM_2_SS_HBMC_0),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_1),  RD_RESP_PS1(0, HBM_2_SS_HBMC_1),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_2),  RD_RESP_PS1(0, HBM_2_SS_HBMC_2),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_3),  RD_RESP_PS1(0, HBM_2_SS_HBMC_3),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_4),  RD_RESP_PS1(0, HBM_2_SS_HBMC_4),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_5),  RD_RESP_PS1(0, HBM_2_SS_HBMC_5),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_6),  RD_RESP_PS1(0, HBM_2_SS_HBMC_6),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_7),  RD_RESP_PS1(0, HBM_2_SS_HBMC_7),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_8),  RD_RESP_PS1(0, HBM_2_SS_HBMC_8),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_9),  RD_RESP_PS1(0, HBM_2_SS_HBMC_9),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_10), RD_RESP_PS1(0, HBM_2_SS_HBMC_10),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_11), RD_RESP_PS1(0, HBM_2_SS_HBMC_11),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_12), RD_RESP_PS1(0, HBM_2_SS_HBMC_12),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_13), RD_RESP_PS1(0, HBM_2_SS_HBMC_13),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_14), RD_RESP_PS1(0, HBM_2_SS_HBMC_14),
        RD_RESP_PS0(0, HBM_2_SS_HBMC_15), RD_RESP_PS1(0, HBM_2_SS_HBMC_15),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_0),  RD_RESP_PS1(0, HBM_3_SS_HBMC_0),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_1),  RD_RESP_PS1(0, HBM_3_SS_HBMC_1),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_2),  RD_RESP_PS1(0, HBM_3_SS_HBMC_2),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_3),  RD_RESP_PS1(0, HBM_3_SS_HBMC_3),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_4),  RD_RESP_PS1(0, HBM_3_SS_HBMC_4),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_5),  RD_RESP_PS1(0, HBM_3_SS_HBMC_5),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_6),  RD_RESP_PS1(0, HBM_3_SS_HBMC_6),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_7),  RD_RESP_PS1(0, HBM_3_SS_HBMC_7),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_8),  RD_RESP_PS1(0, HBM_3_SS_HBMC_8),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_9),  RD_RESP_PS1(0, HBM_3_SS_HBMC_9),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_10), RD_RESP_PS1(0, HBM_3_SS_HBMC_10),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_11), RD_RESP_PS1(0, HBM_3_SS_HBMC_11),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_12), RD_RESP_PS1(0, HBM_3_SS_HBMC_12),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_13), RD_RESP_PS1(0, HBM_3_SS_HBMC_13),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_14), RD_RESP_PS1(0, HBM_3_SS_HBMC_14),
        RD_RESP_PS0(0, HBM_3_SS_HBMC_15), RD_RESP_PS1(0, HBM_3_SS_HBMC_15),
    };
  } else {
    return {
        RD_RESP_PS0(1, HBM_0_SS_HBMC_0),  RD_RESP_PS1(1, HBM_0_SS_HBMC_0),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_1),  RD_RESP_PS1(1, HBM_0_SS_HBMC_1),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_2),  RD_RESP_PS1(1, HBM_0_SS_HBMC_2),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_3),  RD_RESP_PS1(1, HBM_0_SS_HBMC_3),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_4),  RD_RESP_PS1(1, HBM_0_SS_HBMC_4),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_5),  RD_RESP_PS1(1, HBM_0_SS_HBMC_5),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_6),  RD_RESP_PS1(1, HBM_0_SS_HBMC_6),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_7),  RD_RESP_PS1(1, HBM_0_SS_HBMC_7),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_8),  RD_RESP_PS1(1, HBM_0_SS_HBMC_8),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_9),  RD_RESP_PS1(1, HBM_0_SS_HBMC_9),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_10), RD_RESP_PS1(1, HBM_0_SS_HBMC_10),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_11), RD_RESP_PS1(1, HBM_0_SS_HBMC_11),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_12), RD_RESP_PS1(1, HBM_0_SS_HBMC_12),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_13), RD_RESP_PS1(1, HBM_0_SS_HBMC_13),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_14), RD_RESP_PS1(1, HBM_0_SS_HBMC_14),
        RD_RESP_PS0(1, HBM_0_SS_HBMC_15), RD_RESP_PS1(1, HBM_0_SS_HBMC_15),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_0),  RD_RESP_PS1(1, HBM_1_SS_HBMC_0),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_1),  RD_RESP_PS1(1, HBM_1_SS_HBMC_1),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_2),  RD_RESP_PS1(1, HBM_1_SS_HBMC_2),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_3),  RD_RESP_PS1(1, HBM_1_SS_HBMC_3),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_4),  RD_RESP_PS1(1, HBM_1_SS_HBMC_4),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_5),  RD_RESP_PS1(1, HBM_1_SS_HBMC_5),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_6),  RD_RESP_PS1(1, HBM_1_SS_HBMC_6),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_7),  RD_RESP_PS1(1, HBM_1_SS_HBMC_7),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_8),  RD_RESP_PS1(1, HBM_1_SS_HBMC_8),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_9),  RD_RESP_PS1(1, HBM_1_SS_HBMC_9),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_10), RD_RESP_PS1(1, HBM_1_SS_HBMC_10),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_11), RD_RESP_PS1(1, HBM_1_SS_HBMC_11),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_12), RD_RESP_PS1(1, HBM_1_SS_HBMC_12),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_13), RD_RESP_PS1(1, HBM_1_SS_HBMC_13),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_14), RD_RESP_PS1(1, HBM_1_SS_HBMC_14),
        RD_RESP_PS0(1, HBM_1_SS_HBMC_15), RD_RESP_PS1(1, HBM_1_SS_HBMC_15),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_0),  RD_RESP_PS1(1, HBM_2_SS_HBMC_0),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_1),  RD_RESP_PS1(1, HBM_2_SS_HBMC_1),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_2),  RD_RESP_PS1(1, HBM_2_SS_HBMC_2),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_3),  RD_RESP_PS1(1, HBM_2_SS_HBMC_3),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_4),  RD_RESP_PS1(1, HBM_2_SS_HBMC_4),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_5),  RD_RESP_PS1(1, HBM_2_SS_HBMC_5),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_6),  RD_RESP_PS1(1, HBM_2_SS_HBMC_6),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_7),  RD_RESP_PS1(1, HBM_2_SS_HBMC_7),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_8),  RD_RESP_PS1(1, HBM_2_SS_HBMC_8),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_9),  RD_RESP_PS1(1, HBM_2_SS_HBMC_9),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_10), RD_RESP_PS1(1, HBM_2_SS_HBMC_10),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_11), RD_RESP_PS1(1, HBM_2_SS_HBMC_11),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_12), RD_RESP_PS1(1, HBM_2_SS_HBMC_12),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_13), RD_RESP_PS1(1, HBM_2_SS_HBMC_13),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_14), RD_RESP_PS1(1, HBM_2_SS_HBMC_14),
        RD_RESP_PS0(1, HBM_2_SS_HBMC_15), RD_RESP_PS1(1, HBM_2_SS_HBMC_15),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_0),  RD_RESP_PS1(1, HBM_3_SS_HBMC_0),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_1),  RD_RESP_PS1(1, HBM_3_SS_HBMC_1),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_2),  RD_RESP_PS1(1, HBM_3_SS_HBMC_2),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_3),  RD_RESP_PS1(1, HBM_3_SS_HBMC_3),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_4),  RD_RESP_PS1(1, HBM_3_SS_HBMC_4),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_5),  RD_RESP_PS1(1, HBM_3_SS_HBMC_5),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_6),  RD_RESP_PS1(1, HBM_3_SS_HBMC_6),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_7),  RD_RESP_PS1(1, HBM_3_SS_HBMC_7),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_8),  RD_RESP_PS1(1, HBM_3_SS_HBMC_8),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_9),  RD_RESP_PS1(1, HBM_3_SS_HBMC_9),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_10), RD_RESP_PS1(1, HBM_3_SS_HBMC_10),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_11), RD_RESP_PS1(1, HBM_3_SS_HBMC_11),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_12), RD_RESP_PS1(1, HBM_3_SS_HBMC_12),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_13), RD_RESP_PS1(1, HBM_3_SS_HBMC_13),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_14), RD_RESP_PS1(1, HBM_3_SS_HBMC_14),
        RD_RESP_PS0(1, HBM_3_SS_HBMC_15), RD_RESP_PS1(1, HBM_3_SS_HBMC_15),
    };
  }
}

std::vector<uint64_t> GetTpuv7xHbmWriteCounters(int die) {
  if (die == 0) {
    return {
        WR_REQ_PS0(0, HBM_0_SS_HBMC_0),  WR_REQ_PS1(0, HBM_0_SS_HBMC_0),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_1),  WR_REQ_PS1(0, HBM_0_SS_HBMC_1),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_2),  WR_REQ_PS1(0, HBM_0_SS_HBMC_2),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_3),  WR_REQ_PS1(0, HBM_0_SS_HBMC_3),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_4),  WR_REQ_PS1(0, HBM_0_SS_HBMC_4),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_5),  WR_REQ_PS1(0, HBM_0_SS_HBMC_5),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_6),  WR_REQ_PS1(0, HBM_0_SS_HBMC_6),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_7),  WR_REQ_PS1(0, HBM_0_SS_HBMC_7),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_8),  WR_REQ_PS1(0, HBM_0_SS_HBMC_8),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_9),  WR_REQ_PS1(0, HBM_0_SS_HBMC_9),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_10), WR_REQ_PS1(0, HBM_0_SS_HBMC_10),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_11), WR_REQ_PS1(0, HBM_0_SS_HBMC_11),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_12), WR_REQ_PS1(0, HBM_0_SS_HBMC_12),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_13), WR_REQ_PS1(0, HBM_0_SS_HBMC_13),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_14), WR_REQ_PS1(0, HBM_0_SS_HBMC_14),
        WR_REQ_PS0(0, HBM_0_SS_HBMC_15), WR_REQ_PS1(0, HBM_0_SS_HBMC_15),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_0),  WR_REQ_PS1(0, HBM_1_SS_HBMC_0),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_1),  WR_REQ_PS1(0, HBM_1_SS_HBMC_1),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_2),  WR_REQ_PS1(0, HBM_1_SS_HBMC_2),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_3),  WR_REQ_PS1(0, HBM_1_SS_HBMC_3),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_4),  WR_REQ_PS1(0, HBM_1_SS_HBMC_4),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_5),  WR_REQ_PS1(0, HBM_1_SS_HBMC_5),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_6),  WR_REQ_PS1(0, HBM_1_SS_HBMC_6),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_7),  WR_REQ_PS1(0, HBM_1_SS_HBMC_7),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_8),  WR_REQ_PS1(0, HBM_1_SS_HBMC_8),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_9),  WR_REQ_PS1(0, HBM_1_SS_HBMC_9),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_10), WR_REQ_PS1(0, HBM_1_SS_HBMC_10),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_11), WR_REQ_PS1(0, HBM_1_SS_HBMC_11),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_12), WR_REQ_PS1(0, HBM_1_SS_HBMC_12),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_13), WR_REQ_PS1(0, HBM_1_SS_HBMC_13),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_14), WR_REQ_PS1(0, HBM_1_SS_HBMC_14),
        WR_REQ_PS0(0, HBM_1_SS_HBMC_15), WR_REQ_PS1(0, HBM_1_SS_HBMC_15),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_0),  WR_REQ_PS1(0, HBM_2_SS_HBMC_0),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_1),  WR_REQ_PS1(0, HBM_2_SS_HBMC_1),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_2),  WR_REQ_PS1(0, HBM_2_SS_HBMC_2),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_3),  WR_REQ_PS1(0, HBM_2_SS_HBMC_3),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_4),  WR_REQ_PS1(0, HBM_2_SS_HBMC_4),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_5),  WR_REQ_PS1(0, HBM_2_SS_HBMC_5),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_6),  WR_REQ_PS1(0, HBM_2_SS_HBMC_6),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_7),  WR_REQ_PS1(0, HBM_2_SS_HBMC_7),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_8),  WR_REQ_PS1(0, HBM_2_SS_HBMC_8),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_9),  WR_REQ_PS1(0, HBM_2_SS_HBMC_9),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_10), WR_REQ_PS1(0, HBM_2_SS_HBMC_10),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_11), WR_REQ_PS1(0, HBM_2_SS_HBMC_11),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_12), WR_REQ_PS1(0, HBM_2_SS_HBMC_12),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_13), WR_REQ_PS1(0, HBM_2_SS_HBMC_13),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_14), WR_REQ_PS1(0, HBM_2_SS_HBMC_14),
        WR_REQ_PS0(0, HBM_2_SS_HBMC_15), WR_REQ_PS1(0, HBM_2_SS_HBMC_15),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_0),  WR_REQ_PS1(0, HBM_3_SS_HBMC_0),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_1),  WR_REQ_PS1(0, HBM_3_SS_HBMC_1),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_2),  WR_REQ_PS1(0, HBM_3_SS_HBMC_2),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_3),  WR_REQ_PS1(0, HBM_3_SS_HBMC_3),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_4),  WR_REQ_PS1(0, HBM_3_SS_HBMC_4),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_5),  WR_REQ_PS1(0, HBM_3_SS_HBMC_5),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_6),  WR_REQ_PS1(0, HBM_3_SS_HBMC_6),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_7),  WR_REQ_PS1(0, HBM_3_SS_HBMC_7),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_8),  WR_REQ_PS1(0, HBM_3_SS_HBMC_8),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_9),  WR_REQ_PS1(0, HBM_3_SS_HBMC_9),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_10), WR_REQ_PS1(0, HBM_3_SS_HBMC_10),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_11), WR_REQ_PS1(0, HBM_3_SS_HBMC_11),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_12), WR_REQ_PS1(0, HBM_3_SS_HBMC_12),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_13), WR_REQ_PS1(0, HBM_3_SS_HBMC_13),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_14), WR_REQ_PS1(0, HBM_3_SS_HBMC_14),
        WR_REQ_PS0(0, HBM_3_SS_HBMC_15), WR_REQ_PS1(0, HBM_3_SS_HBMC_15),
    };
  } else {
    return {
        WR_REQ_PS0(1, HBM_0_SS_HBMC_0),  WR_REQ_PS1(1, HBM_0_SS_HBMC_0),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_1),  WR_REQ_PS1(1, HBM_0_SS_HBMC_1),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_2),  WR_REQ_PS1(1, HBM_0_SS_HBMC_2),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_3),  WR_REQ_PS1(1, HBM_0_SS_HBMC_3),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_4),  WR_REQ_PS1(1, HBM_0_SS_HBMC_4),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_5),  WR_REQ_PS1(1, HBM_0_SS_HBMC_5),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_6),  WR_REQ_PS1(1, HBM_0_SS_HBMC_6),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_7),  WR_REQ_PS1(1, HBM_0_SS_HBMC_7),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_8),  WR_REQ_PS1(1, HBM_0_SS_HBMC_8),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_9),  WR_REQ_PS1(1, HBM_0_SS_HBMC_9),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_10), WR_REQ_PS1(1, HBM_0_SS_HBMC_10),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_11), WR_REQ_PS1(1, HBM_0_SS_HBMC_11),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_12), WR_REQ_PS1(1, HBM_0_SS_HBMC_12),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_13), WR_REQ_PS1(1, HBM_0_SS_HBMC_13),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_14), WR_REQ_PS1(1, HBM_0_SS_HBMC_14),
        WR_REQ_PS0(1, HBM_0_SS_HBMC_15), WR_REQ_PS1(1, HBM_0_SS_HBMC_15),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_0),  WR_REQ_PS1(1, HBM_1_SS_HBMC_0),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_1),  WR_REQ_PS1(1, HBM_1_SS_HBMC_1),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_2),  WR_REQ_PS1(1, HBM_1_SS_HBMC_2),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_3),  WR_REQ_PS1(1, HBM_1_SS_HBMC_3),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_4),  WR_REQ_PS1(1, HBM_1_SS_HBMC_4),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_5),  WR_REQ_PS1(1, HBM_1_SS_HBMC_5),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_6),  WR_REQ_PS1(1, HBM_1_SS_HBMC_6),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_7),  WR_REQ_PS1(1, HBM_1_SS_HBMC_7),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_8),  WR_REQ_PS1(1, HBM_1_SS_HBMC_8),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_9),  WR_REQ_PS1(1, HBM_1_SS_HBMC_9),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_10), WR_REQ_PS1(1, HBM_1_SS_HBMC_10),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_11), WR_REQ_PS1(1, HBM_1_SS_HBMC_11),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_12), WR_REQ_PS1(1, HBM_1_SS_HBMC_12),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_13), WR_REQ_PS1(1, HBM_1_SS_HBMC_13),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_14), WR_REQ_PS1(1, HBM_1_SS_HBMC_14),
        WR_REQ_PS0(1, HBM_1_SS_HBMC_15), WR_REQ_PS1(1, HBM_1_SS_HBMC_15),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_0),  WR_REQ_PS1(1, HBM_2_SS_HBMC_0),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_1),  WR_REQ_PS1(1, HBM_2_SS_HBMC_1),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_2),  WR_REQ_PS1(1, HBM_2_SS_HBMC_2),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_3),  WR_REQ_PS1(1, HBM_2_SS_HBMC_3),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_4),  WR_REQ_PS1(1, HBM_2_SS_HBMC_4),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_5),  WR_REQ_PS1(1, HBM_2_SS_HBMC_5),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_6),  WR_REQ_PS1(1, HBM_2_SS_HBMC_6),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_7),  WR_REQ_PS1(1, HBM_2_SS_HBMC_7),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_8),  WR_REQ_PS1(1, HBM_2_SS_HBMC_8),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_9),  WR_REQ_PS1(1, HBM_2_SS_HBMC_9),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_10), WR_REQ_PS1(1, HBM_2_SS_HBMC_10),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_11), WR_REQ_PS1(1, HBM_2_SS_HBMC_11),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_12), WR_REQ_PS1(1, HBM_2_SS_HBMC_12),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_13), WR_REQ_PS1(1, HBM_2_SS_HBMC_13),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_14), WR_REQ_PS1(1, HBM_2_SS_HBMC_14),
        WR_REQ_PS0(1, HBM_2_SS_HBMC_15), WR_REQ_PS1(1, HBM_2_SS_HBMC_15),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_0),  WR_REQ_PS1(1, HBM_3_SS_HBMC_0),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_1),  WR_REQ_PS1(1, HBM_3_SS_HBMC_1),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_2),  WR_REQ_PS1(1, HBM_3_SS_HBMC_2),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_3),  WR_REQ_PS1(1, HBM_3_SS_HBMC_3),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_4),  WR_REQ_PS1(1, HBM_3_SS_HBMC_4),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_5),  WR_REQ_PS1(1, HBM_3_SS_HBMC_5),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_6),  WR_REQ_PS1(1, HBM_3_SS_HBMC_6),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_7),  WR_REQ_PS1(1, HBM_3_SS_HBMC_7),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_8),  WR_REQ_PS1(1, HBM_3_SS_HBMC_8),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_9),  WR_REQ_PS1(1, HBM_3_SS_HBMC_9),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_10), WR_REQ_PS1(1, HBM_3_SS_HBMC_10),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_11), WR_REQ_PS1(1, HBM_3_SS_HBMC_11),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_12), WR_REQ_PS1(1, HBM_3_SS_HBMC_12),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_13), WR_REQ_PS1(1, HBM_3_SS_HBMC_13),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_14), WR_REQ_PS1(1, HBM_3_SS_HBMC_14),
        WR_REQ_PS0(1, HBM_3_SS_HBMC_15), WR_REQ_PS1(1, HBM_3_SS_HBMC_15),
    };
  }
}

#undef HBM_PREFIX
#undef RD_RESP_PS0
#undef RD_RESP_PS1
#undef WR_REQ_PS0
#undef WR_REQ_PS1

void ComputeTpuv7GenericBandwidthUtilization(
    const TpuCounterUtil& counters,
    const Tpuv7GenericUtilizationOptions& options, int core,
    UtilizationCounters* utilization) {
  uint64_t hbm_rd_beats = 0;
  uint64_t hbm_wr_beats = 0;

  if (options.is_tpu7) {
    // TODO: Implement for TPU v7 when counter IDs are available.
  } else {
    for (uint64_t counter_id : GetTpuv7xHbmReadCounters(core)) {
      hbm_rd_beats += counters.GetValue(counter_id);
    }
    for (uint64_t counter_id : GetTpuv7xHbmWriteCounters(core)) {
      hbm_wr_beats += counters.GetValue(counter_id);
    }
  }

  const int kHbmBytesPerBeat = 32;
  uint64_t hbm_rd_bytes = hbm_rd_beats * kHbmBytesPerBeat;
  uint64_t hbm_wr_bytes = hbm_wr_beats * kHbmBytesPerBeat;

  uint64_t cycle_addr = 0;
  // NOLINTBEGIN
  if (options.is_tpu7) {
    cycle_addr = TpuCounterIdsTpu7::
        VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;
  } else {
    if (core == 0) {
      cycle_addr = TpuCounterIdsTpu7x::
          VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;
    } else {
      cycle_addr = TpuCounterIdsTpu7x::
          VF_CHIP_DIE1_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;
    }
  }
  // NOLINTEND
  const uint64_t cycles = counters.GetValue(cycle_addr);
  if (cycles == 0) return;

  const double time_s =
      tsl::profiler::CyclesToSeconds(cycles, options.frequency_hz);

  // TODO: Fetch from device properties
  const double peak_hbm_bw_Bps = options.peak_hbm_bw_bps;
  const double peak_bytes = peak_hbm_bw_Bps * time_s;

  AddUtilization(counters, /*node_id=*/0,
                 absl::StrCat("HBM Rd+Wr - core ", core),
                 hbm_rd_bytes + hbm_wr_bytes, peak_bytes, kBytes, utilization);

  const double total_hbm_bytes = hbm_rd_bytes + hbm_wr_bytes;
  if (total_hbm_bytes > 0) {
    AddUtilization(counters, core, "HBM Read Ratio", hbm_rd_bytes,
                   total_hbm_bytes, kPercent, utilization);
    AddUtilization(counters, core, "HBM Write Ratio", hbm_wr_bytes,
                   total_hbm_bytes, kPercent, utilization);
  }
}

void ComputeTpuv7GenericIciBandwidthUtilization(
    const TpuCounterUtil& counters,
    const Tpuv7GenericUtilizationOptions& options,
    UtilizationCounters* utilization) {
  constexpr size_t KIciBytesPerFlit = 128;
  uint64_t ici_rd_bytes = 0;
  uint64_t ici_wr_bytes = 0;

  std::vector<uint64_t> ici_rd_flits_perf_counters;
  std::vector<uint64_t> ici_wr_flits_perf_counters;

  if (options.is_tpu7) {
    // TODO: Implement for TPU v7
  } else {
    // NOLINTBEGIN
    ici_rd_flits_perf_counters = {
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_0_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_1_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_2_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_3_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_RD_REQ_0_INSERTION_COUNT,
    };
    ici_wr_flits_perf_counters = {
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_0_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_WR_REQ_0_INSERTION_COUNT,
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_1_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_WR_REQ_0_INSERTION_COUNT,
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_2_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_WR_REQ_0_INSERTION_COUNT,
        TpuCounterIdsTpu7x::
            VF_CHIP_DIE1_OCI_ICR_3_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_WR_REQ_0_INSERTION_COUNT,
    };
    // NOLINTEND
  }

  uint64_t ici_rd_flits = 0;
  for (uint64_t counter_id : ici_rd_flits_perf_counters) {
    ici_rd_flits += counters.GetValue(counter_id);
  }
  ici_rd_bytes = ici_rd_flits * KIciBytesPerFlit;

  uint64_t ici_wr_flits = 0;
  for (uint64_t counter_id : ici_wr_flits_perf_counters) {
    ici_wr_flits += counters.GetValue(counter_id);
  }
  ici_wr_bytes = ici_wr_flits * KIciBytesPerFlit;

  uint64_t cycle_counter_address = 0;
  // NOLINTBEGIN
  if (options.is_tpu7) {
    cycle_counter_address = TpuCounterIdsTpu7::
        VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;
  } else {
    cycle_counter_address = TpuCounterIdsTpu7x::
        VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;
  }
  // NOLINTEND

  const uint64_t cycles = counters.GetValue(cycle_counter_address);
  if (cycles == 0) return;

  const double time_s =
      tsl::profiler::CyclesToSeconds(cycles, 1000000000);  // Approximation

  const double kPeakIciRdBwBytesPerSecond = 294947368421.0526;
  const double peak_ici_rd_bytes = kPeakIciRdBwBytesPerSecond * time_s;
  const double peak_ici_wr_bytes = peak_ici_rd_bytes;

  AddUtilization(counters, 0, "ICI (Read)", ici_rd_bytes, peak_ici_rd_bytes,
                 kBytes, utilization);
  AddUtilization(counters, 0, "ICI (Write)", ici_wr_bytes, peak_ici_wr_bytes,
                 kBytes, utilization);
}

}  // namespace xprof
