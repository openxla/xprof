// To run the benchmarks in this file using benchy:
//
// 1. Run a specific benchmark (e.g., BM_ConvertXSpaceToTraceEventsContainer):
//    benchy
//      //third_party/xprof/convert:xspace_to_trace_container_benchmark_test \
//    --benchmark_filter="BM_ConvertXSpaceToTraceEventsContainer"
//
// 2. Run all benchmarks in this file:
//    benchy \
//      //third_party/xprof/convert:xspace_to_trace_container_benchmark_test
//
// 3. Compare a specific benchmark against the client's baseline (e.g., "head"):
//    benchy --reference=srcfs \
//      //third_party/xprof/convert:xspace_to_trace_container_benchmark_test \
//      --benchmark_filter="BM_ConvertXSpaceToTraceEventsContainer"
//
// 4. Run a specific benchmark in the Chamber environment for lower
//    noise/variance:
//    benchy --chamber \
//      //third_party/xprof/convert:xspace_to_trace_container_benchmark_test \
//      --benchmark_filter="BM_ConvertXSpaceToTraceEventsContainer"
//    (Note: Acquiring Chamber resources can sometimes be slow or fail
//      depending on availability.)
//
// 5. For more options, see go/benchy and go/chamber.

#include <stdlib.h>
#include <string>
#include "xprof/convert/xplane_to_trace_container.h"

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "testing/base/public/benchmark.h"
#include "absl/log/check.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

void BM_ConvertXSpaceToTraceEventsContainer(benchmark::State& state) {
  std::string xplane_path = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/xprof/convert/test_xplanes/"
      "gpu_training_2.xplane.pb");

  XPlane real_plane;
  CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), xplane_path, &real_plane));

  XSpace space;
  *space.add_planes() = real_plane;

  for (auto s : state) {
    TraceEventsContainer container;
    ConvertXSpaceToTraceEventsContainer("localhost", space, &container);
  }
}
BENCHMARK(BM_ConvertXSpaceToTraceEventsContainer);

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
