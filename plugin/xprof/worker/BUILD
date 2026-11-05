load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

exports_files([
    "grpc_server.cc",
    "grpc_server.h",
    "grpc_utils.cc",
    "grpc_utils.h",
    "stub_factory.cc",
    "stub_factory.h",
    "worker_service.cc",
    "worker_service.h",
])

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
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "//xprof/convert:profile_processor_factory",
        "//xprof/convert:tool_options",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@grpc//:grpc++",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla/pjrt:semaphore",
    ],
)

cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cc"],
    hdrs = ["grpc_server.h"],
    deps = [
        ":worker_service",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@grpc//:grpc++",
        "@grpc//:grpc_security_base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_utils",
    srcs = ["grpc_utils.cc"],
    hdrs = ["grpc_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@grpc//:grpc++",
    ],
)

cc_library(
    name = "stub_factory",
    srcs = ["stub_factory.cc"],
    hdrs = ["stub_factory.h"],
    deps = [
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@grpc//:grpc++",
        "@grpc//:grpc_security_base",
    ],
    alwayslink = 1,
)

cc_test(
    name = "stub_factory_test",
    srcs = ["stub_factory_test.cc"],
    deps = [
        ":stub_factory",
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@com_google_googletest//:gtest_main",
        "@grpc//:grpc++",
    ],
)
