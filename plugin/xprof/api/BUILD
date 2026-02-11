# Description:
#   XProf APIs that could be called

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "continuous_profiling_snapshot",
    srcs = ["continuous_profiling_snapshot.py"],
    deps = [
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

py_test(
    name = "continuous_profiling_snapshot_test",
    srcs = ["continuous_profiling_snapshot_test.py"],
    deps = [
        ":continuous_profiling_snapshot",
        "@absl_py//absl/testing:absltest",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)
