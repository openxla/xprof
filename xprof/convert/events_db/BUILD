# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("//plugin/xprof/build_utils/oss:build_config.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//xprof:license"],
    default_visibility = ["@org_xprof//xprof/convert/events_db:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "schema_test",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parent_event_tracker",
    srcs = ["parent_event_tracker.cc"],
    hdrs = ["parent_event_tracker.h"],
    deps = [
        ":event_utils",
        ":record_consumer",
        ":schema",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "parent_event_tracker_test",
    srcs = ["parent_event_tracker_test.cc"],
    deps = [
        ":event_utils",
        ":parent_event_tracker",
        ":record_consumer",
        ":schema",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "event_utils",
    srcs = ["event_utils.cc"],
    hdrs = ["event_utils.h"],
    deps = [
        ":schema",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_xprof//xprof/convert:dcn_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "event_utils_test",
    srcs = ["event_utils_test.cc"],
    deps = [
        ":event_utils",
        ":schema",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "record_consumer",
    hdrs = ["record_consumer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "custom_trace_parser",
    srcs = ["custom_trace_parser.cc"],
    hdrs = ["custom_trace_parser.h"],
    deps = [
        ":event_utils",
        ":record_consumer",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "custom_trace_parser_test",
    srcs = ["custom_trace_parser_test.cc"],
    deps = [
        ":custom_trace_parser",
        ":event_utils",
        ":record_consumer",
        ":schema",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
    ],
)

cc_library(
    name = "host_trace_parser",
    srcs = ["host_trace_parser.cc"],
    hdrs = ["host_trace_parser.h"],
    deps = [
        ":event_utils",
        ":record_consumer",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "host_trace_parser_test",
    srcs = ["host_trace_parser_test.cc"],
    deps = [
        ":event_utils",
        ":host_trace_parser",
        ":record_consumer",
        ":schema",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
    ],
)

cc_library(
    name = "gpu_trace_parser",
    srcs = ["gpu_trace_parser.cc"],
    hdrs = ["gpu_trace_parser.h"],
    deps = [
        ":event_utils",
        ":record_consumer",
        ":schema",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/utils:gpu_event_stats",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@org_xprof//xprof/utils:performance_info_wrapper",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "gpu_trace_parser_test",
    srcs = ["gpu_trace_parser_test.cc"],
    deps = [
        ":event_utils",
        ":gpu_trace_parser",
        ":record_consumer",
        ":schema",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//xprof/utils:hlo_cost_analysis_wrapper",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@org_xprof//xprof/utils:xprof_gpu_cost_analysis",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
    ],
)
