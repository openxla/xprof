#include "xprof/convert/processors/memory_profile_processor.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/processors/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_memory_profile.h"

namespace xprof {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

absl::Status MemoryProfileProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  if (session_snapshot.XSpaceSize() != 1) {
    return tsl::errors::InvalidArgument(
        "Memory profile tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  LOG(INFO) << "Processing memory profile for host: "
            << session_snapshot.GetHostname(0);

  std::string memory_profile_json;
  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(0, &arena));
  PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                             /*derived_timeline=*/false);
  TF_RETURN_IF_ERROR(
      ConvertXSpaceToMemoryProfileJson(*xspace, &memory_profile_json));

  SetOutput(memory_profile_json, "application/json");
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("memory_profile", MemoryProfileProcessor);

}  // namespace xprof
