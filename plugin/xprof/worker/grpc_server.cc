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
#include "grpc/grpc.h"
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

void InitializeGrpcServer(int port, int num_cqs, int min_pollers,
                          int max_pollers) {
  std::string server_address = absl::StrCat(kServerAddressPrefix, port);
  ::grpc::ServerBuilder builder;
  // Set options for the synchronous server to tune performance.
  // NUM_CQS: Number of completion queues to use for handling requests.
  // MIN_POLLERS: Minimum number of threads per completion queue.
  // MAX_POLLERS: Maximum number of threads per completion queue.
  // gRPC's sync server uses a thread pool to service incoming requests, with
  // threads polling one or more completion queues for work. These settings
  // control the number of queues and the number of threads polling each queue.
  // Increasing pollers or CQs can improve performance for highly concurrent
  // workloads, but consumes more threads and memory.
  // See:
  // https://grpc.github.io/grpc/cpp/classgrpc_1_1_server_builder.html#aff66bd93cba7d4240a64550fe1fca88d
  builder.SetSyncServerOption(::grpc::ServerBuilder::NUM_CQS, num_cqs);
  builder.SetSyncServerOption(::grpc::ServerBuilder::MIN_POLLERS, min_pollers);
  builder.SetSyncServerOption(::grpc::ServerBuilder::MAX_POLLERS, max_pollers);
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  // The period in milliseconds after which a keepalive ping is sent on the
  // transport.
  // For more details, see:
  // https://github.com/grpc/grpc/blob/master/doc/keepalive.md
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 20000);
  // The amount of time in milliseconds the sender of the keepalive ping waits
  // for an acknowledgement. If it does not receive an acknowledgment within
  // this time, it will close the connection.
  // For more details, see:
  // https://github.com/grpc/grpc/blob/master/doc/keepalive.md
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
  // The minimum time in milliseconds that gRPC Core would expect between
  // receiving successive pings. If the time between successive pings is less
  // than this time, then the ping will be considered a bad ping from the peer.
  // For more details, see:
  // https://github.com/grpc/grpc/blob/master/doc/keepalive.md
  builder.AddChannelArgument(
      GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 10000);
  // The maximum number of pings that can be sent when there is no data/header
  // frame to be sent. Setting it to 0 allows sending pings without such a
  // restriction.
  // For more details, see:
  // https://github.com/grpc/grpc/blob/master/doc/keepalive.md
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  // The maximum number of bad pings that the server will tolerate before
  // sending an HTTP2 GOAWAY frame and closing the transport. Setting it to 0
  // allows the server to accept any number of bad pings.
  // For more details, see:
  // https://github.com/grpc/grpc/blob/master/doc/keepalive.md
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PING_STRIKES, 2);
  // Set the maximum connection age to 50 seconds. This forces connections to be
  // gracefully terminated, causing clients to reconnect and re-resolve DNS,
  // allowing them to discover new pods in under a minute.
  // For more details, see:
  // https://github.com/grpc/proposal/blob/master/A9-server-side-conn-mgt.md
  builder.AddChannelArgument(GRPC_ARG_MAX_CONNECTION_AGE_MS, 50000);
  // Set the grace period for max connection age to 10 minutes. This allows
  // long-running RPCs (up to 10 minutes) to complete before the connection is
  // forcefully closed.
  // For more details, see:
  // https://github.com/grpc/proposal/blob/master/A9-server-side-conn-mgt.md
  builder.AddChannelArgument(GRPC_ARG_MAX_CONNECTION_AGE_GRACE_MS, 600000);
  // Set the maximum message length that the channel can receive to unlimited.
  builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
  // Set the maximum message length that the channel can send to unlimited.
  builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
  worker_service =
      std::make_unique<::xprof::profiler::ProfileWorkerServiceImpl>();
  builder.RegisterService(worker_service.get());
  server = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << server_address;
}

}  // namespace profiler
}  // namespace xprof
