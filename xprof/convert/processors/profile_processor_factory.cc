// Copyright 2024 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
#include "xprof/convert/processors/profile_processor_factory.h"

#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/processors/profile_processor.h"
#include "xprof/convert/tool_options.h"

namespace xprof {

ProfileProcessorFactory& ProfileProcessorFactory::GetInstance() {
  static auto* instance = new ProfileProcessorFactory();
  return *instance;
}

void ProfileProcessorFactory::Register(
    absl::string_view tool_name,
    absl::AnyInvocable<std::unique_ptr<ProfileProcessor>(
        const tensorflow::profiler::ToolOptions&) const>
        creator) {
  creators_[tool_name] = std::move(creator);
}

std::unique_ptr<ProfileProcessor> ProfileProcessorFactory::Create(
    absl::string_view tool_name,
    const tensorflow::profiler::ToolOptions& options) const {
  auto it = creators_.find(tool_name);
  if (it == creators_.end()) {
    LOG(ERROR) << "No ProfileProcessor registered for tool: " << tool_name;
    return nullptr;
  }
  return it->second(options);
}

RegisterProfileProcessor::RegisterProfileProcessor(
    absl::string_view tool_name,
    absl::AnyInvocable<std::unique_ptr<ProfileProcessor>(
        const tensorflow::profiler::ToolOptions&) const>
        creator) {
  ProfileProcessorFactory::GetInstance().Register(tool_name,
                                                  std::move(creator));
}

}  // namespace xprof
