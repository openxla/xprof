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

#ifndef THIRD_PARTY_XPROF_UTILS_XPLANE_HLO_FIXER_H_
#define THIRD_PARTY_XPROF_UTILS_XPLANE_HLO_FIXER_H_

#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xprof {

// Fixes XSpace data by updating HLO Proto metadata and IDs.
void FixHloMetadataInXSpace(tensorflow::profiler::XSpace* space);

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_UTILS_XPLANE_HLO_FIXER_H_
