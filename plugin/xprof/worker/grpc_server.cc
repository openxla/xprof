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

#include "plugin/xprof/worker/grpc_server.h"

#include <memory>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "plugin/xprof/worker/worker_service.h"

namespace xprof {
namespace profiler {

constexpr std::string_view kServerAddressPrefix = "0.0.0.0:";

static std::unique_ptr<::grpc::Server> server;
static std::unique_ptr<::xprof::profiler::ProfileWorkerServiceImpl>
    worker_service;

void InitializeGrpcServer(int port) {
  std::string server_address = absl::StrCat(kServerAddressPrefix, port);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  worker_service =
      std::make_unique<::xprof::profiler::ProfileWorkerServiceImpl>();
  builder.RegisterService(worker_service.get());
  server = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << server_address;
}

}  // namespace profiler
}  // namespace xprof
