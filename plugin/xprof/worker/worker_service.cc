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

#include "plugin/xprof/worker/worker_service.h"

#include <memory>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tsl/platform/host_info.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/worker/grpc_utils.h"

namespace xprof {
namespace profiler {

::grpc::Status ProfileWorkerServiceImpl::GetProfileData(
    ::grpc::ServerContext* context,
    const ::xprof::pywrap::WorkerProfileDataRequest* request,
    ::xprof::pywrap::WorkerProfileDataResponse* response) {
  const auto& origin_request = request->origin_request();
  LOG(INFO) << "GetProfileData tool:" << origin_request.tool_name()
            << " session:" << origin_request.session_id() << " params:{"
            << absl::StrJoin(origin_request.parameters(), ",",
                             absl::PairFormatter("="))
            << "}";

  if (!semaphore_.TryAcquire(1)) {
    LOG(INFO) << "GetProfileData tool:" << origin_request.tool_name()
              << " session:" << origin_request.session_id()
              << " is waiting in queue as max concurrent "
              << "requests limit reached.";
    semaphore_.Acquire(1);
  }
  LOG(INFO) << "GetProfileData tool:" << origin_request.tool_name()
            << " session:" << origin_request.session_id()
            << " acquired the semaphore and is starting processing.";
  auto cleanup = absl::MakeCleanup([this]() { semaphore_.Release(1); });

  tensorflow::profiler::ToolOptions tool_options;
  for (const auto& [key, value] : origin_request.parameters()) {
    tool_options[key] = value;
  }
  auto processor = xprof::ProfileProcessorFactory::GetInstance().Create(
      origin_request.tool_name(), tool_options);
  if (!processor) {
    LOG(ERROR) << "GetProfileData failed: Can not find tool: "
               << origin_request.tool_name();
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "Can not find tool: " + origin_request.tool_name());
  }

  absl::StatusOr<std::string> map_output_file =
      processor->Map(origin_request.session_id());
  if (!map_output_file.ok()) {
    LOG(ERROR) << "GetProfileData failed with status: "
               << map_output_file.status();
    return ToGrpcStatus(map_output_file.status());
  }
  response->set_output(*map_output_file);
  response->set_worker_id(tsl::port::Hostname());

  LOG(INFO)
      << "ProfileWorkerServiceImpl::GetProfileData finished successfully by "
         "worker: "
      << response->worker_id();
  return ::grpc::Status::OK;
}

}  // namespace profiler
}  // namespace xprof
