// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
#ifndef THIRD_PARTY_XPROF_CONVERT_UNIFIED_PROFILE_PROCESSOR_FACTORY_H_
#define THIRD_PARTY_XPROF_CONVERT_UNIFIED_PROFILE_PROCESSOR_FACTORY_H_

#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"

namespace xprof {

class UnifiedProfileProcessorFactory {
 public:
  using Creator = absl::AnyInvocable<std::unique_ptr<UnifiedProfileProcessor>(
      const tensorflow::profiler::ToolOptions&) const>;

  static UnifiedProfileProcessorFactory& GetInstance();

  void Register(absl::string_view tool_name, Creator creator);

  std::unique_ptr<UnifiedProfileProcessor> Create(
      absl::string_view tool_name,
      const tensorflow::profiler::ToolOptions& options) const;

 private:
  UnifiedProfileProcessorFactory() = default;

  friend class absl::NoDestructor<UnifiedProfileProcessorFactory>;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Creator> creators_ ABSL_GUARDED_BY(mu_);
};

// Registration macro.
class RegisterUnifiedProfileProcessor {
 public:
  RegisterUnifiedProfileProcessor(
      absl::string_view tool_name,
      UnifiedProfileProcessorFactory::Creator creator);
};

#define REGISTER_UNIFIED_PROFILE_PROCESSOR(unique_id, tool_name, ClassName)    \
  ABSL_ATTRIBUTE_UNUSED static const ::xprof::RegisterUnifiedProfileProcessor \
      register_##unique_id(                                                  \
          tool_name, [](const tensorflow::profiler::ToolOptions& options) {  \
            return std::make_unique<ClassName>(options);                     \
          });

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_UNIFIED_PROFILE_PROCESSOR_FACTORY_H_
