load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["@org_xprof//plugin/xprof/cli:__subpackages__"])

pytype_strict_library(
    name = "upload_trace_tool",
    srcs = ["upload_trace_tool.py"],
    deps = ["@org_xprof//plugin/xprof/cli/internal/oss:xprof_client"],
)
