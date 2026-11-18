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

#ifndef XPROF_CONVERT_XPLANE_TO_LLO_H_
#define XPROF_CONVERT_XPLANE_TO_LLO_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"

namespace tensorflow {
namespace profiler {

// Compiled TPU kernels can register their LLO (Low-Level Optimizer) modules
// with the profiler, which stores each module as a `StatType::kLloProto` stat
// on an event metadata of the host metadata plane of the XSpace. The Kernel
// Viewer reads only those modules, so this file caches them per host in a
// small LLO-only XPlane (`<host>.llo_xplane.pb` in the session run directory)
// that later requests read instead of the whole XSpace.

// File name suffix of the per-host LLO XPlane cache files.
inline constexpr absl::string_view kLloXPlaneSuffix = ".llo_xplane.pb";

// Returns whether `xspace` carries LLO modules.
bool HasLloProtoMetadata(const XSpace& xspace);

// Copies the event metadata of the host metadata plane of `xspace` that carry
// an LLO module, together with the stat metadata they refer to, into
// `llo_plane`. Returns the number of modules copied.
int ExtractLloXPlane(const XSpace& xspace, XPlane* llo_plane);

// Writes the LLO XPlane of `xspace`, the XSpace of `host_name`, to the cache
// file of the host if the session has an accessible run directory and the file
// does not exist yet.
absl::Status MaybeCacheLloXPlane(const SessionSnapshot& session_snapshot,
                                 absl::string_view host_name,
                                 const XSpace& xspace);

// Returns the LLO XPlane of `host_name` (the first host of the session if
// empty): from its cache file, which is written on first use, or straight from
// the XSpace if the session has no accessible run directory. The plane has no
// event metadata if the host has no LLO modules.
absl::StatusOr<XPlane> GetLloXPlaneByHostName(
    const SessionSnapshot& session_snapshot, absl::string_view host_name);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_LLO_H_
