#include "xprof/utils/hlo_proto_to_module.h"

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
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
              operand_ids: 6
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
              operand_ids: 6
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

  const auto& module_proto = hlo_proto.hlo_module();
  ASSERT_OK_AND_ASSIGN(auto module,
                       ConvertHloModuleProtoToModule(module_proto));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  // Check that ids are consecutive
  EXPECT_THAT(module->entry_computation()->instructions(),
              ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                          Property(&xla::HloInstruction::local_id, 1),
                          Property(&xla::HloInstruction::local_id, 2)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
