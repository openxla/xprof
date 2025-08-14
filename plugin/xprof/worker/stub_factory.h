/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_STUB_FACTORY_H_
#define THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_STUB_FACTORY_H_

#include <memory>
#include <string>

#include "plugin/xprof/protobuf/worker_service.grpc.pb.h"

namespace xprof {
namespace profiler {

// Initializes the stubs with the given worker service addresses.
// This must be called once before calling GetNextStub().
void InitializeStubs(const std::string& worker_service_addresses);

// Returns the next stub in a round-robin fashion.
std::shared_ptr<xprof::pywrap::grpc::XprofAnalysisWorkerService::Stub>
GetNextStub();

}  // namespace profiler
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_STUB_FACTORY_H_
