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

#include "xprof/convert/kernel_viewer_registry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

absl::StatusOr<std::string> EchoKernelName(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  return absl::StrCat(
      session_snapshot.GetHostname(0), ":",
      GetParamWithDefault<std::string>(options, "kernel_name", "none"));
}

absl::StatusOr<SessionSnapshot> CreateSession() {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::make_unique<XSpace>());
  return SessionSnapshot::Create({"ram://in_memory/host_a.xplane.pb"},
                                 std::move(xspaces));
}

TEST(KernelViewerRegistryTest, ForwardsToTheRegisteredHandler) {
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session, CreateSession());
  const ToolOptions options = {{"kernel_name", std::string("fusion.1")}};

  RegisterKernelViewerHandler(nullptr);
  EXPECT_FALSE(IsKernelViewerRegistered());
  EXPECT_THAT(ConvertSessionToKernelViewer(session, options),
              StatusIs(absl::StatusCode::kUnimplemented));

  RegisterKernelViewerHandler(&EchoKernelName);
  EXPECT_TRUE(IsKernelViewerRegistered());
  EXPECT_THAT(ConvertSessionToKernelViewer(session, options),
              IsOkAndHolds("host_a:fusion.1"));

  RegisterKernelViewerHandler(nullptr);
  EXPECT_FALSE(IsKernelViewerRegistered());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
