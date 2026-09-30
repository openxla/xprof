#include "xprof/utils/hlo_proto_to_module.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "google/protobuf/text_format.h"
#include "xla/hlo/ir/hlo_instruction.h"

using ::testing::ElementsAre;
using ::testing::Property;

namespace tensorflow {
namespace profiler {
namespace {

TEST(HloProtoToModuleTest, FixNonConsecutiveInstructionIds) {
  xla::HloProto hlo_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        hlo_module {
          name: "some_module"
          entry_computation_name: "some_module"
          computations {
            name: "some_module"
            instructions {
              name: "arg0.1"
              opcode: "parameter"
              shape {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
              id: 4294967297
            }
            instructions {
              name: "arg1.1"
              opcode: "parameter"
              shape {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
              parameter_number: 1
              id: 4294967298
            }
            instructions {
              name: "XLA_Retvals.1"
              opcode: "tuple"
              shape {
                element_type: TUPLE
                tuple_shapes {
                  element_type: S32
                  layout { tail_padding_alignment_in_elements: 1 }
                }
              }
              id: 4294967303
              operand_ids: 1
            }
            id: 1
            root_id: 4294967303
          }
          host_program_shape {
            parameters {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            parameters {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            result {
              element_type: TUPLE
              tuple_shapes {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
            }
            parameter_names: "arg0"
            parameter_names: "arg1"
          }
          id: 1
          entry_computation_id: 1
        }
      )pb",
      &hlo_proto));

  ASSERT_OK_AND_ASSIGN(auto module, ConvertHloProtoToModule(hlo_proto));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  // Check that ids are consecutive
  EXPECT_THAT(module->entry_computation()->instructions(),
              ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                          Property(&xla::HloInstruction::local_id, 1),
                          Property(&xla::HloInstruction::local_id, 2)));
  // Check correct operand translation
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->name(),
            "arg0.1");
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->local_id(),
            0);
  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operands(),
      ElementsAre(module->entry_computation()->parameter_instruction(0)));
}

TEST(HloProtoToModuleTest, FixNonConsecutiveInstructionIdsForModule) {
  xla::HloProto hlo_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        hlo_module {
          name: "some_module"
          entry_computation_name: "some_module"
          computations {
            name: "some_module"
            instructions {
              name: "arg0.1"
              opcode: "parameter"
              shape {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
              id: 4294967297
            }
            instructions {
              name: "arg1.1"
              opcode: "parameter"
              shape {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
              parameter_number: 1
              id: 4294967298
            }
            instructions {
              name: "XLA_Retvals.1"
              opcode: "tuple"
              shape {
                element_type: TUPLE
                tuple_shapes {
                  element_type: S32
                  layout { tail_padding_alignment_in_elements: 1 }
                }
              }
              id: 4294967303
              operand_ids: 1
            }
            id: 1
            root_id: 4294967303
          }
          host_program_shape {
            parameters {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            parameters {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            result {
              element_type: TUPLE
              tuple_shapes {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
            }
            parameter_names: "arg0"
            parameter_names: "arg1"
          }
          id: 1
          entry_computation_id: 1
        }
      )pb",
      &hlo_proto));


  ASSERT_OK_AND_ASSIGN(auto module,
                       ConvertHloProtoToModule(hlo_proto));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  // Check that ids are consecutive
  EXPECT_THAT(module->entry_computation()->instructions(),
              ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                          Property(&xla::HloInstruction::local_id, 1),
                          Property(&xla::HloInstruction::local_id, 2)));
  // Check correct operand translation
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->name(),
            "arg0.1");
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->local_id(),
            0);
  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operands(),
      ElementsAre(module->entry_computation()->parameter_instruction(0)));
}

TEST(HloProtoToModuleTest, MultipleComputationsCallingCallee) {
  xla::HloProto hlo_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        hlo_module {
          name: "module_with_callees"
          entry_computation_name: "entry"
          entry_computation_id: 3
          id: 100
          host_program_shape {
            parameters {
              element_type: F32
              dimensions: [10]
              layout { tail_padding_alignment_in_elements: 1 }
            }
            result {
              element_type: F32
              dimensions: [10]
              layout { tail_padding_alignment_in_elements: 1 }
            }
            parameter_names: "x"
          }
          computations {
            name: "callee"
            id: 1
            instructions {
              name: "p0"
              opcode: "parameter"
              shape { element_type: F32 }
              id: 10
            }
            instructions {
              name: "p1"
              opcode: "parameter"
              parameter_number: 1
              shape { element_type: F32 }
              id: 11
            }
            instructions {
              name: "add"
              opcode: "add"
              shape { element_type: F32 }
              operand_ids: 10
              operand_ids: 11
              id: 12
            }
            root_id: 12
          }
          computations {
            name: "subcomp"
            id: 2
            instructions {
              name: "sub_p0"
              opcode: "parameter"
              shape { element_type: F32 dimensions: [10] }
              id: 20
            }
            instructions {
              name: "sub_p1"
              opcode: "parameter"
              parameter_number: 1
              shape { element_type: F32 dimensions: [10] }
              id: 21
            }
            instructions {
              name: "map"
              opcode: "map"
              shape { element_type: F32 dimensions: [10] }
              operand_ids: 20
              operand_ids: 21
              called_computation_ids: 1
              id: 22
            }
            root_id: 22
          }
          computations {
            name: "entry"
            id: 3
            instructions {
              name: "entry_p0"
              opcode: "parameter"
              shape { element_type: F32 dimensions: [10] }
              id: 30
            }
            instructions {
              name: "call_sub"
              opcode: "call"
              shape { element_type: F32 dimensions: [10] }
              operand_ids: 30
              operand_ids: 30
              called_computation_ids: 2
              id: 31
            }
            instructions {
              name: "map2"
              opcode: "map"
              shape { element_type: F32 dimensions: [10] }
              operand_ids: 30
              operand_ids: 31
              called_computation_ids: 1
              id: 32
            }
            root_id: 32
          }
        }
      )pb",
      &hlo_proto));

  ASSERT_OK_AND_ASSIGN(auto module, ConvertHloProtoToModule(hlo_proto));
  EXPECT_NE(module, nullptr);
  EXPECT_EQ(module->computation_count(), 3);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
