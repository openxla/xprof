/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_UNIFIED_TOOLS_REGISTRATION_H_
#define THIRD_PARTY_XPROF_CONVERT_UNIFIED_TOOLS_REGISTRATION_H_

#include "absl/status/status.h"

namespace xprof {
// Registers unified profile processors for various profiling tools (e.g.,
// hlo_stats, memory_viewer) with the processor factory.
void RegisterUnifiedToolRegistrations();

// Registers unified tools and verifies that every expected tool can create a
// processor. Returns FailedPrecondition if any expected tool is missing.
// Expected tools: hlo_stats, memory_viewer, op_profile, overview_page.
absl::Status EnsureUnifiedToolsRegistered();
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_UNIFIED_TOOLS_REGISTRATION_H_
