load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin:__subpackages__"])

pytype_strict_library(
    name = "xprof_data",
    srcs = ["xprof_data.py"],
    deps = [
        ":decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "decorators",
    srcs = ["decorators.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = ["@absl_py//absl/logging"],
)
