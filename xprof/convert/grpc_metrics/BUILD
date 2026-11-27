# Parses the gRPC and TCP flow events of the DCN Analyzer tool (grpc_metrics).

# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "json_utils",
    srcs = ["json_utils.cc"],
    hdrs = ["json_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "transfer_key_parser",
    srcs = ["transfer_key_parser.cc"],
    hdrs = ["transfer_key_parser.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grpc_metrics_utils",
    srcs = [
        "database_populator.cc",
        "grpc_metrics_utils.cc",
        "xspace_extractor.cc",
    ],
    hdrs = [
        "database_populator.h",
        "grpc_metrics_types.h",
        "grpc_metrics_utils.h",
        "xspace_extractor.h",
    ],
    deps = [
        ":json_utils",
        ":transfer_key_parser",
        "//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "//plugin/xprof/protobuf:grpc_metrics_proto_cc",
        "//plugin/xprof/protobuf:topology_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@org_xprof//xprof/convert:xspace_to_dcn_slack_analysis",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "json_utils_test",
    srcs = ["json_utils_test.cc"],
    deps = [
        ":json_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "transfer_key_parser_test",
    srcs = ["transfer_key_parser_test.cc"],
    deps = [
        ":transfer_key_parser",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "database_populator_test",
    srcs = ["database_populator_test.cc"],
    deps = [
        ":grpc_metrics_utils",
        "//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "//plugin/xprof/protobuf:grpc_metrics_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "xspace_extractor_test",
    srcs = ["xspace_extractor_test.cc"],
    deps = [
        ":grpc_metrics_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
    ],
)

cc_test(
    name = "grpc_metrics_utils_test",
    srcs = ["grpc_metrics_utils_test.cc"],
    deps = [
        ":grpc_metrics_utils",
        "//plugin/xprof/protobuf:grpc_metrics_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
