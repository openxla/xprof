# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//xprof:__subpackages__"])

cc_library(
    name = "zstd_compression",
    srcs = ["zstd_compression.cc"],
    hdrs = ["zstd_compression.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@net_zstd//:zstd",
    ],
)

cc_test(
    name = "zstd_compression_test",
    srcs = ["zstd_compression_test.cc"],
    deps = [
        ":zstd_compression",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "trace_data_to_compressed_delta_series_proto",
    srcs = ["trace_data_to_compressed_delta_series_proto.cc"],
    hdrs = ["trace_data_to_compressed_delta_series_proto.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":zstd_compression",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_raw_proto_cc",
        "@org_xprof//xprof/convert/trace_viewer:trace_events",
        "@org_xprof//xprof/convert/trace_viewer:trace_events_to_json",
        "@tsl//tsl/profiler/lib:context_types",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "trace_data_to_compressed_delta_series_proto_test",
    srcs = ["trace_data_to_compressed_delta_series_proto_test.cc"],
    deps = [
        ":trace_data_to_compressed_delta_series_proto",
        ":zstd_compression",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@org_xprof//xprof/convert/trace_viewer:trace_events",
    ],
)
