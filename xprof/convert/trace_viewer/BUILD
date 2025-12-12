# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

default_visibility = [
    "//perftools/gputools/profiler/collector:__pkg__",
    "@org_tensorflow//tensorflow/core/profiler:internal",
    "//xprof:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "trace_events_filter_interface",
    hdrs = ["trace_events_filter_interface.h"],
    deps = [
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "trace_viewer_visibility",
    srcs = [
        "trace_viewer_visibility.cc",
    ],
    hdrs = ["trace_viewer_visibility.h"],
    deps = [
        ":trace_events_filter_interface",
        ":trace_options",
        ":trace_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_test(
    name = "trace_viewer_visibility_test",
    srcs = ["trace_viewer_visibility_test.cc"],
    deps = [
        ":trace_options",
        ":trace_viewer_visibility",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "trace_viewer_color",
    srcs = ["trace_viewer_color.cc"],
    hdrs = ["trace_viewer_color.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
    ],
)

cc_library(
    name = "trace_events_to_json",
    srcs = ["trace_events_to_json.cc"],
    hdrs = ["trace_events_to_json.h"],
    deps = [
        ":trace_events_util",
        ":trace_viewer_color",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_googlesource_code_re2//:re2",
        "@org_xprof//plugin/xprof/protobuf:task_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_raw_proto_cc",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/lib:context_types",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
    ],
)

cc_library(
    name = "trace_event_arguments_builder",
    hdrs = ["trace_event_arguments_builder.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:trace_events_raw_proto_cc",
    ],
)

cc_library(
    name = "trace_events_util",
    srcs = ["trace_events_util.cc"],
    hdrs = ["trace_events_util.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_test(
    name = "trace_events_util_test",
    srcs = ["trace_events_util_test.cc"],
    deps = [
        ":trace_events_util",
        "//base:types",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "trace_events",
    srcs = ["trace_events.cc"],
    hdrs = ["trace_events.h"],
    deps = [
        ":prefix_trie",
        ":trace_events_filter_interface",
        ":trace_events_util",
        ":trace_viewer_visibility",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_xprof//plugin/xprof/protobuf:task_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_raw_proto_cc",
        "@org_xprof//xprof/convert:xprof_thread_pool_executor",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@xla//xla/tsl/lib/io:block",
        "@xla//xla/tsl/lib/io:iterator",
        "@xla//xla/tsl/lib/io:table",
        "@xla//xla/tsl/lib/io:table_options",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "trace_options",
    srcs = ["trace_options.cc"],
    hdrs = ["trace_options.h"],
    deps = [
        ":trace_events_filter_interface",
        ":trace_events_to_json",
        ":trace_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@org_xprof//plugin/xprof/protobuf:trace_filter_config_proto_cc",
        "@org_xprof//xprof/convert:tool_options",
    ],
)

cc_test(
    name = "trace_options_test",
    srcs = ["trace_options_test.cc"],
    deps = [
        ":trace_events_filter_interface",
        ":trace_options",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//xprof/convert:tool_options",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
    ],
)

cc_library(
    name = "prefix_trie",
    srcs = ["prefix_trie.cc"],
    hdrs = ["prefix_trie.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:prefix_trie_node_proto_cc",
        "@xla//xla/tsl/lib/io:block",
        "@xla//xla/tsl/lib/io:iterator",
        "@xla//xla/tsl/lib/io:table",
        "@xla//xla/tsl/lib/io:table_options",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_test(
    name = "trace_utils_test",
    srcs = ["trace_utils_test.cc"],
    deps = [
        ":trace_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "prefix_trie_test",
    srcs = ["prefix_trie_test.cc"],
    deps = [
        ":prefix_trie",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:env",
    ],
)
