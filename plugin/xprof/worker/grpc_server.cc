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
