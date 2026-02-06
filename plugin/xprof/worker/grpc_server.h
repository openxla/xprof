/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_SERVER_H_
#define THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_SERVER_H_

namespace xprof {
namespace profiler {
// TODO: b/442301153 - Add ShutdownGrpcServer() as well.
void InitializeGrpcServer(int port, int max_concurrent_requests);

}  // namespace profiler
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_SERVER_H_
