// To run the benchmarks in this file using benchy:
//
// 1. Run a specific benchmark (e.g., BM_PreprocessSingleHostXSpace):
//    benchy //third_party/xprof/convert:preprocess_single_host_xplane_test \
//      --benchmark_filter="BM_PreprocessSingleHostXSpace"
//
// 2. Run all benchmarks in this file:
//    benchy //third_party/xprof/convert:preprocess_single_host_xplane_test
//
// 3. Compare a specific benchmark against the client's baseline (e.g., "head"):
//    benchy --reference=srcfs \
//      //third_party/xprof/convert:preprocess_single_host_xplane_test \
//      --benchmark_filter="BM_PreprocessSingleHostXSpace"
//
// 4. Run a specific benchmark in the Chamber environment for lower
//    noise/variance:
//    benchy --chamber \
//      //third_party/xprof/convert:preprocess_single_host_xplane_test \
//      --benchmark_filter="BM_PreprocessSingleHostXSpace"
//    (Note: Acquiring Chamber resources can sometimes be slow or fail
//      depending on availability.)
//
// 5. For more options, see go/benchy and go/chamber.

#include "xprof/convert/preprocess_single_host_xplane.h"

#include <string>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "testing/base/public/benchmark.h"
#include "absl/log/check.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
