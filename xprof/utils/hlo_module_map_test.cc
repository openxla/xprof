/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xprof/utils/hlo_module_map.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::ElementsAre;

TEST(HloInstructionWrapperTest, ExtractsTensorsAndFingerprint) {
  absl::string_view hlo_text = R"hlo(
    HloModule test_module
    ENTRY test {
      x = f32[64,64]{1,0} parameter(0)
      y = f32[64,64]{1,0} parameter(1)
      add = f32[64,64]{1,0} add(x, y)
      ROOT dot = f32[64,64]{1,0} dot(x, add), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })hlo";
  ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));
  const xla::HloInstruction* dot =
      hlo_module->entry_computation()->root_instruction();
  const xla::HloInstruction* add = dot->operand(1);

  HloInstructionWrapper dot_wrapper(dot);
  HloInstructionWrapper add_wrapper(add);

  EXPECT_THAT(dot_wrapper.InputTensors(),
              ElementsAre("f32[64,64]", "f32[64,64]"));
  EXPECT_THAT(dot_wrapper.OutputTensors(), ElementsAre("f32[64,64]"));
  EXPECT_NE(dot_wrapper.Fingerprint(), 0);

  EXPECT_THAT(add_wrapper.InputTensors(),
              ElementsAre("f32[64,64]", "f32[64,64]"));
  EXPECT_THAT(add_wrapper.OutputTensors(), ElementsAre("f32[64,64]"));
  EXPECT_NE(add_wrapper.Fingerprint(), 0);
  EXPECT_NE(dot_wrapper.Fingerprint(), add_wrapper.Fingerprint());
}

TEST(HloInstructionWrapperTest, ExtractsTupleOutputTensors) {
  absl::string_view hlo_text = R"hlo(
    HloModule test_module
    ENTRY test {
      x = f32[10]{0} parameter(0)
      y = f32[20]{0} parameter(1)
      ROOT tuple = (f32[10]{0}, f32[20]{0}) tuple(x, y)
    })hlo";
  ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));
  const xla::HloInstruction* tuple_inst =
      hlo_module->entry_computation()->root_instruction();

  HloInstructionWrapper tuple_wrapper(tuple_inst);

  EXPECT_THAT(tuple_wrapper.InputTensors(), ElementsAre("f32[10]", "f32[20]"));
  EXPECT_THAT(tuple_wrapper.OutputTensors(), ElementsAre("(f32[10], f32[20])"));
}

TEST(ProcessHloModuleMapFromXSpaceTest,
     InvokesStatefulCostAnalysisFactorySequentiallyAcrossMultipleModules) {
  XSpace space;
  XPlane* metadata_plane = space.add_planes();
  metadata_plane->set_name(tsl::profiler::kMetadataPlaneName);
  tsl::profiler::XPlaneBuilder builder(metadata_plane);
  tsl::profiler::XStatMetadata* hlo_proto_stat =
      builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloProto));
  tsl::profiler::XStatMetadata* program_id_stat =
      builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId));

  constexpr int kNumModules = 8;
  for (int i = 1; i <= kNumModules; ++i) {
    std::string hlo_text = absl::StrCat("HloModule mod_", i, R"hlo(
      ENTRY main {
        x = f32[16,16]{1,0} parameter(0)
        ROOT y = f32[16,16]{1,0} add(x, x)
      })hlo");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> mod,
                         xla::ParseAndReturnUnverifiedModule(hlo_text));
    xla::HloProto proto;
    *proto.mutable_hlo_module() = mod->ToProto();

    tsl::profiler::XEventMetadata* event_metadata =
        builder.GetOrCreateEventMetadata(i);
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> stats(
        event_metadata, &builder);
    stats.AddStatValue(*program_id_stat, static_cast<uint64_t>(100 + i));
    stats.AddStatValue(*hlo_proto_stat, proto);
  }

  // Non-thread-safe stateful vector mutated inside the factory; under TSAN this
  // catches any concurrent invocation of `create_cost_analysis` on worker
  // threads.
  std::vector<int> invocation_log;
  HloCostAnalysisWrapper::Factory factory =
      [&invocation_log]() -> std::unique_ptr<HloCostAnalysisWrapper> {
    invocation_log.push_back(static_cast<int>(invocation_log.size()) + 1);
    return nullptr;
  };

  HloModuleMap hlo_module_map;
  ProcessHloModuleMapFromXSpace(hlo_module_map, &space, factory);

  EXPECT_EQ(invocation_log.size(), kNumModules);
  ASSERT_EQ(hlo_module_map.size(), kNumModules);
  for (int i = 1; i <= kNumModules; ++i) {
    auto it = hlo_module_map.find(static_cast<uint64_t>(100 + i));
    ASSERT_NE(it, hlo_module_map.end());
    EXPECT_EQ(it->second.Name(), absl::StrCat("mod_", i));
  }
}

TEST(ProcessHloModuleMapFromXSpaceTest,
     PopulatesProgramIdToNameMapEvenWhenOneModuleFailsConversion) {
  XSpace space;
  XPlane* metadata_plane = space.add_planes();
  metadata_plane->set_name(tsl::profiler::kMetadataPlaneName);
  tsl::profiler::XPlaneBuilder builder(metadata_plane);
  tsl::profiler::XStatMetadata* hlo_proto_stat =
      builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloProto));
  tsl::profiler::XStatMetadata* program_id_stat =
      builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId));

  // Module 1: valid HLO module.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> valid_mod,
      xla::ParseAndReturnUnverifiedModule(R"hlo(
        HloModule valid_module
        ENTRY main {
          x = f32[8]{0} parameter(0)
          ROOT y = f32[8]{0} add(x, x)
        })hlo"));
  xla::HloProto valid_proto;
  *valid_proto.mutable_hlo_module() = valid_mod->ToProto();
  {
    tsl::profiler::XEventMetadata* meta = builder.GetOrCreateEventMetadata(1);
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> stats(meta,
                                                                      &builder);
    stats.AddStatValue(*program_id_stat, uint64_t{201});
    stats.AddStatValue(*hlo_proto_stat, valid_proto);
  }

  // Module 2: malformed HloProto (has module name, no computations ->
  // ConvertHloProtoToModule fails).
  xla::HloProto malformed_proto;
  malformed_proto.mutable_hlo_module()->set_name("malformed_module");
  {
    tsl::profiler::XEventMetadata* meta = builder.GetOrCreateEventMetadata(2);
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> stats(meta,
                                                                      &builder);
    stats.AddStatValue(*program_id_stat, uint64_t{202});
    stats.AddStatValue(*hlo_proto_stat, malformed_proto);
  }

  HloCostAnalysisWrapper::Factory factory = []() { return nullptr; };
  HloModuleMap hlo_module_map;
  tsl::protobuf::Map<uint64_t, std::string> program_id_to_name_map;
  ProcessHloModuleMapFromXSpace(hlo_module_map, &space, factory,
                                &program_id_to_name_map);

  EXPECT_EQ(hlo_module_map.size(), 1);
  EXPECT_TRUE(hlo_module_map.contains(201));
  EXPECT_FALSE(hlo_module_map.contains(202));
  ASSERT_EQ(program_id_to_name_map.size(), 2);
  EXPECT_EQ(program_id_to_name_map.at(201), "valid_module");
  EXPECT_EQ(program_id_to_name_map.at(202), "malformed_module");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
