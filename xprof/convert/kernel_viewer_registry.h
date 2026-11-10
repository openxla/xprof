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

#ifndef XPROF_CONVERT_KERNEL_VIEWER_REGISTRY_H_
#define XPROF_CONVERT_KERNEL_VIEWER_REGISTRY_H_

#include <string>

#include "absl/status/statusor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {

// The Kernel Viewer tool renders the statically scheduled LLO modules of
// compiled TPU kernels. Its converter is only available in builds that embed
// the TPU ISA definitions, which register it here at static initialization
// time; other builds do not advertise the tool.

// Serves a Kernel Viewer request (see the `request_type`, `kernel_name`,
// `hlo_module` and `host_name` options) as a JSON string.
using KernelViewerHandler = absl::StatusOr<std::string> (*)(
    const SessionSnapshot& session_snapshot, const ToolOptions& options);

// Registers the handler of the "kernel_viewer" tool. The last registration
// wins.
void RegisterKernelViewerHandler(KernelViewerHandler handler);

// Returns whether a Kernel Viewer handler is registered.
bool IsKernelViewerRegistered();

// Serves a Kernel Viewer request with the registered handler, or returns an
// UNIMPLEMENTED error if none is registered.
absl::StatusOr<std::string> ConvertSessionToKernelViewer(
    const SessionSnapshot& session_snapshot, const ToolOptions& options);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_KERNEL_VIEWER_REGISTRY_H_
