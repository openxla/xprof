// To run the benchmarks in this file using benchy:
//
// 1. Run a specific benchmark (e.g., BM_ConvertXSpaceToOpStats):
//    benchy //third_party/xprof/convert:xplane_to_op_stats_benchmark_test \
//      --benchmark_filter="BM_ConvertXSpaceToOpStats"
//
// 2. Run all benchmarks in this file:
//    benchy //third_party/xprof/convert:xplane_to_op_stats_benchmark_test
//
// 3. Compare a specific benchmark against the client's baseline (e.g., "head"):
//    benchy --reference=srcfs \
//      //third_party/xprof/convert:xplane_to_op_stats_benchmark_test \
//      --benchmark_filter="BM_ConvertXSpaceToOpStats"
//
// 4. Run a specific benchmark in the Chamber environment for lower
//    noise/variance:
//    benchy --chamber \
//      //third_party/xprof/convert:xplane_to_op_stats_benchmark_test \
//      --benchmark_filter="BM_ConvertXSpaceToOpStats"
//    (Note: Acquiring Chamber resources can sometimes be slow or fail
//      depending on availability.)
//
// 5. For more options, see go/benchy and go/chamber.

#include <string>
#include "xprof/convert/xplane_to_op_stats.h"

#include "absl/log/check.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
