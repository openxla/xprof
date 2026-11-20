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

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpc/grpc.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "plugin/xprof/worker/worker_service.h"

namespace xprof {
namespace profiler {
namespace {

constexpr char kWorkerHostEnvVar[] = "XPROF_WORKER_HOST";
constexpr absl::string_view kDefaultServerHost = "127.0.0.1";
constexpr int kMaxReceiveMessageLength = 4 * 1024 * 1024;

ABSL_CONST_INIT absl::Mutex server_mutex(absl::kConstInit);
absl::NoDestructor<std::unique_ptr<::grpc::Server>> server
    ABSL_GUARDED_BY(server_mutex);
absl::NoDestructor<std::unique_ptr<::xprof::profiler::ProfileWorkerServiceImpl>>
    worker_service ABSL_GUARDED_BY(server_mutex);

void ShutdownLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(server_mutex) {
  if (*server != nullptr) {
    (*server)->Shutdown();
    (*server)->Wait();
    server->reset();
  }
  worker_service->reset();
}

int StartGrpcServerImpl(int port, int max_concurrent_requests,
                        absl::string_view logdir,
                        std::optional<absl::string_view> host_override) {
  absl::MutexLock lock(server_mutex);
  ShutdownLocked();

  std::string effective_host;
  if (host_override.has_value()) {
    effective_host = std::string(*host_override);
  } else if (const char* env_host = std::getenv(kWorkerHostEnvVar);
             env_host != nullptr) {
    effective_host = env_host;
  }
  std::string server_address =
      internal::FormatGrpcServerAddress(effective_host, port);
  ::grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials(),
                           &selected_port);
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
  // Bound incoming metadata request size to prevent memory exhaustion DoS.
  builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH,
                             kMaxReceiveMessageLength);
  // Set the maximum message length that the channel can send to unlimited.
  builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
  auto new_worker_service =
      std::make_unique<::xprof::profiler::ProfileWorkerServiceImpl>(
          max_concurrent_requests, logdir);
  builder.RegisterService(new_worker_service.get());
  std::unique_ptr<::grpc::Server> new_server = builder.BuildAndStart();
  if (new_server == nullptr || selected_port <= 0) {
    if (new_server != nullptr) {
      new_server->Shutdown();
      new_server->Wait();
    }
    LOG(ERROR) << "Failed to start gRPC worker server on " << server_address;
    return -1;
  }
  *worker_service = std::move(new_worker_service);
  *server = std::move(new_server);
  LOG(INFO) << "Server listening on " << server_address << " (bound port "
            << selected_port << ") with max_concurrent_requests "
            << max_concurrent_requests;
  return selected_port;
}

}  // namespace

namespace internal {

std::string FormatGrpcServerAddress(absl::string_view host, int port) {
  host = absl::StripAsciiWhitespace(host);
  absl::string_view effective_host = host.empty() ? kDefaultServerHost : host;
  if (absl::StrContains(effective_host, ':') &&
      !absl::StartsWith(effective_host, "[")) {
    return absl::StrCat("[", effective_host, "]:", port);
  }
  return absl::StrCat(effective_host, ":", port);
}

int InitializeGrpcServerForTesting(
    int port, int max_concurrent_requests, absl::string_view logdir,
    std::optional<absl::string_view> host_override) {
  return StartGrpcServerImpl(port, max_concurrent_requests, logdir,
                             host_override);
}

void ShutdownGrpcServerForTesting() {
  absl::MutexLock lock(server_mutex);
  ShutdownLocked();
}

}  // namespace internal

void InitializeGrpcServer(int port, int max_concurrent_requests,
                          absl::string_view logdir) {
  StartGrpcServerImpl(port, max_concurrent_requests, logdir, std::nullopt);
}

}  // namespace profiler
}  // namespace xprof
