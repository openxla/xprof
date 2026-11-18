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

#include <atomic>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

std::atomic<KernelViewerHandler> kernel_viewer_handler{nullptr};

}  // namespace

void RegisterKernelViewerHandler(KernelViewerHandler handler) {
  kernel_viewer_handler.store(handler, std::memory_order_release);
}

bool IsKernelViewerRegistered() {
  return kernel_viewer_handler.load(std::memory_order_acquire) != nullptr;
}

absl::StatusOr<std::string> ConvertSessionToKernelViewer(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  KernelViewerHandler handler =
      kernel_viewer_handler.load(std::memory_order_acquire);
  if (handler == nullptr) {
    return absl::UnimplementedError(
        "The Kernel Viewer is not available in this build.");
  }
  return handler(session_snapshot, options);
}

}  // namespace profiler
}  // namespace tensorflow
