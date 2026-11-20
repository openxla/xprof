# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

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

cc_library(
    name = "worker_service",
    srcs = ["worker_service.cc"],
    hdrs = ["worker_service.h"],
    deps = [
        ":grpc_utils",
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@org_xprof//xprof/convert:profile_processor_factory",
        "@org_xprof//xprof/convert:tool_options",
        "@tsl//tsl/platform:path",
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
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc_security_base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_test(
    name = "worker_service_test",
    srcs = ["worker_service_test.cc"],
    deps = [
        ":grpc_server",
        ":worker_service",
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc_security_base",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//xprof/convert:file_utils",
        "@org_xprof//xprof/convert:overview_page_processor",
        "@org_xprof//xprof/convert:profile_processor",
        "@org_xprof//xprof/convert:profile_processor_factory",
        "@org_xprof//xprof/convert:repository",
        "@org_xprof//xprof/convert:tool_options",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:env",
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
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc_security_base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_test(
    name = "stub_factory_test",
    srcs = ["stub_factory_test.cc"],
    deps = [
        ":stub_factory",
        "//plugin/xprof/protobuf:worker_service_cc_grpc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)
