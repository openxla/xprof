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

#include "xprof/convert/op_profile_builder.h"

#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "tsl/platform/protobuf.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(OpProfileBuilderTest, ArenaAllocatedNodeSortingNoCrash) {
  tsl::protobuf::Arena arena;
  auto* root = tsl::protobuf::Arena::Create<op_profile::Node>(&arena);

  OpProfileOptions options;
  options.children_per_node = 3;
  options.group_by = OpProfileGrouping::kByCategory;

  OpProfileBuilder builder(options, root);

  for (int i = 0; i < 10; ++i) {
    FlatOpMetrics op;
    op.set_hlo_name(absl::StrCat("op_", i));
    op.set_op_id(i + 1);
    op.set_time_ps((10 - i) * 100);
    op.set_self_time_ps((10 - i) * 100);
    op.set_category("conv");
    op.set_core_type(FlatOpMetrics::TENSOR_CORE);
    op.set_long_name(absl::StrCat("op_", i, "_long"));
    op.mutable_source_info()->set_file_name(absl::StrCat("src_", i, ".cc"));
    op.mutable_source_info()->set_line_number(i * 5);
    op.mutable_source_info()->set_stack_frame(
        absl::StrCat("frame_", i, "_line1\nframe_", i, "_line2"));
    builder.AddOp(op);
  }

  // Finalize sorts and prunes children on arena-allocated root node tree.
  builder.Finalize(
      /*peak_gigaflops_per_second_per_core=*/1000.0,
      /*peak_mem_gibibytes_per_second_per_core=*/{100.0, 100.0, 100.0},
      /*total_time_ps=*/10000);

  ASSERT_EQ(root->children_size(), 1);
  const auto& conv_cat = root->children(0);
  EXPECT_LE(conv_cat.children_size(), 3);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
