# load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//xprof:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_binary(
    name = "xspace_to_parquet",
    srcs = ["xspace_to_parquet.cc"],
    deps = [
        "//third_party/arrow:arrow_core",
        "//third_party/arrow:util_hdrs",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@org_xprof//xprof/convert/events_db:parquet_record_consumer",
        "@org_xprof//xprof/convert/events_db:record_consumer",
        "@org_xprof//xprof/convert/events_db:schema",
        "@org_xprof//xprof/convert/events_db:xspace_parser",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_test(
    name = "xspace_to_parquet_test",
    srcs = ["xspace_to_parquet_test.cc"],
    data = [":xspace_to_parquet"],
    deps = [
        "//file/base:path",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:subprocess",
    ],
)
