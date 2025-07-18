# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "signal_provider",
    hdrs = ["signal_provider.h"],
    deps = [
        ":tool_data_provider",
        "//util/task:status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tpu_input_pipeline_proto_cc",
    ],
)

cc_library(
    name = "smart_suggestion_rule",
    hdrs = ["smart_suggestion_rule.h"],
    deps = [
        ":signal_provider",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_library(
    name = "tool_data_provider",
    hdrs = ["tool_data_provider.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
    ],
)

cc_library(
    name = "tool_data_provider_impl",
    hdrs = ["tool_data_provider_impl.h"],
    deps = [
        ":tool_data_provider",
        "//util/task:status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//xprof/convert:multi_xplanes_to_op_stats",
        "@org_xprof//xprof/convert:op_stats_to_input_pipeline_analysis",
        "@org_xprof//xprof/convert:op_stats_to_overview_page",
        "@org_xprof//xprof/convert:repository",
    ],
)

cc_library(
    name = "input_bound_rule",
    hdrs = ["input_bound_rule.h"],
    deps = [
        ":signal_provider",
        ":smart_suggestion_rule",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_library(
    name = "host_processing_bound_rule",
    hdrs = ["host_processing_bound_rule.h"],
    deps = [
        ":input_bound_rule",
        ":signal_provider",
        ":smart_suggestion_rule",
        "//util/task:status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_library(
    name = "data_transfer_bound_rule",
    hdrs = ["data_transfer_bound_rule.h"],
    deps = [
        ":input_bound_rule",
        ":signal_provider",
        ":smart_suggestion_rule",
        "//util/task:status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_library(
    name = "smart_suggestion_rule_factory",
    hdrs = ["smart_suggestion_rule_factory.h"],
    deps = [
        ":smart_suggestion_rule",
    ],
)

cc_library(
    name = "all_rules",
    hdrs = ["all_rules.h"],
    deps = [
        ":data_transfer_bound_rule",
        ":host_processing_bound_rule",
        ":input_bound_rule",
        ":smart_suggestion_rule_factory",
    ],
)

cc_library(
    name = "smart_suggestion_engine",
    srcs = ["smart_suggestion_engine.cc"],
    hdrs = ["smart_suggestion_engine.h"],
    deps = [
        ":signal_provider",
        ":smart_suggestion_rule",
        ":smart_suggestion_rule_factory",
        "//util/task:status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_test(
    name = "host_processing_bound_rule_test",
    srcs = ["host_processing_bound_rule_test.cc"],
    deps = [
        ":host_processing_bound_rule",
        ":signal_provider",
        ":tool_data_provider",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)

cc_test(
    name = "data_transfer_bound_rule_test",
    srcs = ["data_transfer_bound_rule_test.cc"],
    deps = [
        ":data_transfer_bound_rule",
        ":signal_provider",
        ":tool_data_provider",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:smart_suggestion_proto_cc",
    ],
)
