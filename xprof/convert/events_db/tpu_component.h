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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_TPU_COMPONENT_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_TPU_COMPONENT_H_

namespace xprof::events_db::internal {

// TPU hardware and execution components used as XLine IDs within TPU XPlanes.
//
// NOTE: This enum defines an open-source subset of TPU components. Explicit
// integer values are intentionally preserved to match the canonical internal
// definitions, ensuring backward and forward compatibility when parsing
// serialized `.xplane.pb` / `.xspace.pb` profiles. Gaps between integer
// values correspond to omitted internal-only hardware telemetry lines.
enum TpuComponent {
  kTensorCoreStepCounter = 1,
  kTensorCoreHloModule = 2,
  kTensorCoreHLO = 3,
  kTensorCoreTraceMe = 7,
  kTensorCoreOverlay = 8,
  kTensorCore = 9,
  kSparseCoreModule = 66,
  kSparseCoreOps = 67,
  kSparseCoreSyncs = 68,
  kSparseCoreTecBase = 85,
  kSparseCoreTec15 = 100,
  kSparseCoreStepCounter = 118,
  kSparseCoreOverlay = 143,
};

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_TPU_COMPONENT_H_
