#include "xprof/convert/xplane_to_tools_data_with_profile_processor.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/path.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/worker_service.pb.h"
#include "plugin/xprof/worker/grpc_utils.h"
#include "plugin/xprof/worker/stub_factory.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr absl::string_view kXplaneFileName = ".xplane.pb";

std::string GetXSpaceFilePath(const SessionSnapshot& session_snapshot,
                              const std::string& hostname) {
  return tsl::io::JoinPath(session_snapshot.GetSessionRunDir(),
                           hostname + kXplaneFileName.data());
}

xprof::pywrap::WorkerProfileDataRequest CreateWorkerProfileDataRequest(
    const std::string& xspace_path, const absl::string_view tool_name,
    const ToolOptions& options) {
  ::xprof::pywrap::WorkerProfileDataRequest request;
  request.mutable_origin_request()->set_session_id(xspace_path);
  request.mutable_origin_request()->set_tool_name(std::string(tool_name));
  for (const auto& option : options) {
    const auto& [key, value] = option;
    if (std::holds_alternative<std::string>(value)) {
      request.mutable_origin_request()->mutable_parameters()->insert(
          {key, std::get<std::string>(value)});
    } else if (std::holds_alternative<int>(value)) {
      request.mutable_origin_request()->mutable_parameters()->insert(
          {key, std::to_string(std::get<int>(value))});
    } else if (std::holds_alternative<bool>(value)) {
      request.mutable_origin_request()->mutable_parameters()->insert(
          {key, std::get<bool>(value) ? "true" : "false"});
    }
  }
  return request;
}

absl::StatusOr<std::string> CallWorkerService(const std::string& xspace_path,
                                              const absl::string_view tool_name,
                                              const ToolOptions& options) {
  ::xprof::pywrap::WorkerProfileDataRequest request =
      CreateWorkerProfileDataRequest(xspace_path, tool_name, options);

  auto stub = ::xprof::profiler::GetNextStub();
  if (!stub) {
    return absl::InternalError("No worker service stub available.");
  }

  ::grpc::ClientContext context;
  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status grpc_status =
      stub->GetProfileData(&context, request, &response);

  if (!grpc_status.ok()) {
    LOG(ERROR) << "gRPC call to worker failed with status_code: "
               << grpc_status.error_code()
               << ", error_message: " << grpc_status.error_message()
               << ", error_details: " << grpc_status.error_details()
               << ", request: " << request.DebugString();
    return ::xprof::profiler::ToAbslStatus(grpc_status);
  }
  LOG(INFO) << "gRPC response: tool=" << tool_name
            << ", worker_id=" << response.worker_id()
            << ", session=" << xspace_path;
  return response.output();
}

absl::Status RunMapReduce(const SessionSnapshot& session_snapshot,
                          const absl::string_view tool_name,
                          xprof::ProfileProcessor* processor,
                          const ToolOptions& options) {
  const int num_hosts = session_snapshot.XSpaceSize();
  std::vector<absl::StatusOr<std::string>> map_outputs(num_hosts);

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), __FUNCTION__,
                                        num_hosts);
    for (int i = 0; i < num_hosts; ++i) {
      thread_pool.Schedule([&session_snapshot, &tool_name, &options,
                            &map_outputs, i] {
        std::string hostname = session_snapshot.GetHostname(i);
        std::string xspace_path = GetXSpaceFilePath(session_snapshot, hostname);
        map_outputs[i] = CallWorkerService(xspace_path, tool_name, options);
      });
    }
  }

  std::vector<std::string> map_output_files;
  map_output_files.reserve(num_hosts);
  for (int i = 0; i < num_hosts; ++i) {
    TF_RETURN_IF_ERROR(map_outputs[i].status());
    map_output_files.push_back(*std::move(map_outputs[i]));
  }
  LOG(INFO) << "Started reducing outputs for tool: " << tool_name
            << " num_hosts: " << num_hosts;
  absl::Time start_time = absl::Now();
  absl::Status reduce_status =
      processor->Reduce(session_snapshot, map_output_files);
  absl::Duration reduce_time = absl::Now() - start_time;
  LOG(INFO) << "Finished reducing outputs for tool: " << tool_name
            << " num_hosts: " << num_hosts << " time taken: " << reduce_time;
  return reduce_status;
}

absl::Status ProcessSession(xprof::ProfileProcessor* processor,
                            const SessionSnapshot& session_snapshot,
                            const ToolOptions& options) {
  TF_RETURN_IF_ERROR(processor->ProcessSession(session_snapshot, options));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> ConvertMultiXSpacesToToolDataWithProfileProcessor(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  absl::string_view session_id = session_snapshot.GetSessionRunDir();
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options)
            << " using ProfileProcessor"
            << " session_id: " << session_id;

  absl::Time start_time = absl::Now();

  auto processor =
      xprof::ProfileProcessorFactory::GetInstance().Create(tool_name, options);
  if (!processor) {
    return tsl::errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }

  if (processor->ShouldUseWorkerService(session_snapshot, options)) {
    // This branch is for the Map/Reduce flow, potentially distributed in the
    // future.
    LOG(INFO) << "Using worker service for tool: " << tool_name;
    TF_RETURN_IF_ERROR(
        RunMapReduce(session_snapshot, tool_name, processor.get(), options));
  } else {
    // This branch is for processing the session directly.
    LOG(INFO) << "Using local processing for tool: " << tool_name;
    TF_RETURN_IF_ERROR(
        ProcessSession(processor.get(), session_snapshot, options));
  }

  LOG(INFO) << "Total time for tool " << tool_name << ": "
            << absl::Now() - start_time << " session_id: " << session_id;
  return processor->GetData();
}

}  // namespace profiler
}  // namespace tensorflow
