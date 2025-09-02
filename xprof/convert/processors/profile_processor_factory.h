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
#ifndef THIRD_PARTY_XPROF_CONVERT_PROFILE_PROCESSOR_FACTORY_H_
#define THIRD_PARTY_XPROF_CONVERT_PROFILE_PROCESSOR_FACTORY_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/processors/profile_processor.h"
#include "xprof/convert/tool_options.h"

namespace xprof {

class ProfileProcessorFactory {
 public:
  using Creator = std::unique_ptr<ProfileProcessor>(
      const tensorflow::profiler::ToolOptions& options);

  static ProfileProcessorFactory& GetInstance();

  void Register(absl::string_view tool_name,
                absl::AnyInvocable<std::unique_ptr<ProfileProcessor>(
                    const tensorflow::profiler::ToolOptions&) const>
                    creator);

  std::unique_ptr<ProfileProcessor> Create(
      absl::string_view tool_name,
      const tensorflow::profiler::ToolOptions& options) const;

 private:
  ProfileProcessorFactory() = default;
  absl::flat_hash_map<std::string,
                      absl::AnyInvocable<std::unique_ptr<ProfileProcessor>(
                          const tensorflow::profiler::ToolOptions&) const>>
      creators_;
};

// Registration macro.
class RegisterProfileProcessor {
 public:
  RegisterProfileProcessor(absl::string_view tool_name,
                           absl::AnyInvocable<std::unique_ptr<ProfileProcessor>(
                               const tensorflow::profiler::ToolOptions&) const>
                               creator);
};

#define REGISTER_PROFILE_PROCESSOR(tool_name, ClassName)                    \
  static const auto* const register_##ClassName =                           \
      new ::xprof::RegisterProfileProcessor(                                \
          tool_name, [](const tensorflow::profiler::ToolOptions& options) { \
            return std::make_unique<ClassName>(options);                    \
          });

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_PROFILE_PROCESSOR_FACTORY_H_
