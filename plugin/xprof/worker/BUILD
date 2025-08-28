# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "worker_service",
    srcs = ["worker_service.cc"],
    hdrs = ["worker_service.h"],
    deps = [
        ":grpc_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@org_xprof//xprof/convert:profile_processor_factory",
        "@org_xprof//xprof/convert:tool_options",
    ],
)

cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cc"],
    hdrs = ["grpc_server.h"],
    deps = [
        ":worker_service",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grpc_utils",
    srcs = ["grpc_utils.cc"],
    hdrs = ["grpc_utils.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "stub_factory",
    srcs = ["stub_factory.cc"],
    hdrs = ["stub_factory.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@org_xprof//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
    ],
)
