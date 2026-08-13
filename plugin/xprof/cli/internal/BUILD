load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin/xprof:__subpackages__"])

pytype_strict_library(
    name = "xprof_data",
    srcs = ["xprof_data.py"],
    deps = [
        ":decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/protobuf:op_profile_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "decorators",
    srcs = ["decorators.py"],
    visibility = ["//plugin/xprof:__subpackages__"],
    deps = ["@absl_py//absl/logging"],
)
