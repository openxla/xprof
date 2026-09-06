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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"

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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
