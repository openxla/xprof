/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_GRAPHVIZ_HELPER_H_
#define XPROF_CONVERT_GRAPHVIZ_HELPER_H_

#include <functional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"

namespace tensorflow {
namespace profiler {

inline std::function<absl::StatusOr<std::string>(absl::string_view)>*
    url_renderer = nullptr;

// Convert dot into visual graph in html
std::string WrapDotInHtml(absl::string_view dot,
                          absl::string_view layout_engine = "dot");

// Precondition: (url_renderer != nullptr || format != kUrl).
//
// (We specify this as a precondition rather than checking it in here and
// returning an error because we want to fail quickly when there's no URL
// renderer available, and this function runs only after we've done all the work
// of producing dot for the graph.)
inline absl::Status CheckPrecondition(xla::RenderedGraphFormat format) {
  if (format == xla::RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return absl::FailedPreconditionError(
        "Can't render as URL; no URL renderer was registered.");
  }
  return absl::OkStatus();
}

// Convert dot into certain format
inline absl::StatusOr<std::string> WrapDotInFormat(
    std::string dot, xla::RenderedGraphFormat format) {
  TF_RETURN_IF_ERROR(CheckPrecondition(format));
  switch (format) {
    case xla::RenderedGraphFormat::kUrl:
      if (url_renderer == nullptr) {
        return absl::InternalError("url_renderer is null");
      }
      return (*url_renderer)(dot);
    case xla::RenderedGraphFormat::kHtml:
      return WrapDotInHtml(dot);
    case xla::RenderedGraphFormat::kDot:
      return std::string(dot);
  }
}

// Registers a function which implements RenderedGraphFormat::kUrl.
// The input to the function is dot, and the output should be a URL or an error.
// There can only be one active renderer, and the last call to this function
// wins.
inline void RegisterGraphvizURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view)> renderer) {
  if (url_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterGraphToURLRenderer. Last call "
                    "wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
  }
  delete url_renderer;
  url_renderer =
      new std::function<absl::StatusOr<std::string>(absl::string_view)>(
          std::move(renderer));
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_GRAPHVIZ_HELPER_H_
