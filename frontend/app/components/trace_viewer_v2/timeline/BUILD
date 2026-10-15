# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = [
        "@imgui",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
    ],
)

wasm_cc_library(
    name = "draw_helpers",
    srcs = ["draw_helpers.cc"],
    hdrs = ["draw_helpers.h"],
    deps = [
        ":constants",
        "@imgui",
    ],
)

wasm_cc_library(
    name = "motion",
    srcs = ["motion.cc"],
    hdrs = ["motion.h"],
)

cc_test(
    name = "motion_test",
    srcs = ["motion_test.cc"],
    deps = [
        ":motion",
        "@com_google_googletest//:gtest_main",
    ],
)

wasm_cc_library(
    name = "time_range",
    srcs = ["time_range.cc"],
    hdrs = ["time_range.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@org_xprof//frontend/app/components/trace_viewer_v2:animation",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
    ],
)

cc_test(
    name = "time_range_test",
    srcs = ["time_range_test.cc"],
    deps = [
        ":time_range",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//frontend/app/components/trace_viewer_v2:animation",
    ],
)

wasm_cc_library(
    name = "timeline",
    srcs = [
        "timeline.cc",
        "timeline_hlo_dependencies.cc",
        "timeline_player_sync.cc",
        "timeline_schedule_sync.cc",
    ],
    hdrs = [
        "timeline.h",
        "timeline_hlo_dependencies.h",
    ],
    deps = [
        ":constants",
        ":draw_helpers",
        ":motion",
        ":time_range",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@imgui",
        "@org_xprof//frontend/app/components/trace_viewer_v2:animation",
        "@org_xprof//frontend/app/components/trace_viewer_v2:event_data",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/fonts",
        "@org_xprof//frontend/app/components/trace_viewer_v2/helper:clipboard",
        "@org_xprof//frontend/app/components/trace_viewer_v2/helper:counter_formatter",
        "@org_xprof//frontend/app/components/trace_viewer_v2/helper:time_formatter",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ] + select({
        "@org_xprof//frontend/app/components/trace_viewer_v2:is_emscripten": [
            "@org_xprof//frontend/app/components/trace_viewer_v2:imgui_webgpu_backend",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "timeline_test",
    srcs = ["timeline_test.cc"],
    deps = [
        ":constants",
        ":draw_helpers",
        ":time_range",
        ":timeline",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@imgui",
        "@org_xprof//frontend/app/components/trace_viewer_v2:animation",
        "@org_xprof//frontend/app/components/trace_viewer_v2:event_data",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/fonts",
        "@org_xprof//frontend/app/components/trace_viewer_v2/helper:time_formatter",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_test(
    name = "timeline_hlo_dependencies_test",
    srcs = ["timeline_hlo_dependencies_test.cc"],
    deps = [
        ":constants",
        ":timeline",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//frontend/app/components/trace_viewer_v2:event_data",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
    ],
)

wasm_cc_library(
    name = "track_packer",
    srcs = ["track_packer.cc"],
    hdrs = ["track_packer.h"],
    deps = [
        ":constants",
        ":timeline",
        "//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:proto_event_ref",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_parser_core",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:wire_trace_reader",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_test(
    name = "track_packer_test",
    srcs = ["track_packer_test.cc"],
    deps = [
        ":timeline",
        ":track_packer",
        "//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:proto_event_ref",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:wire_trace_reader",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

wasm_cc_library(
    name = "data_provider",
    srcs = [
        "data_provider.cc",
    ],
    hdrs = ["data_provider.h"],
    deps = [
        ":constants",
        ":time_range",
        ":timeline",
        ":track_packer",
        "//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@com_googlesource_code_re2//:re2",
        "@imgui",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/helper:time_formatter",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:proto_event_ref",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_packer",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_parser_core",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_tree",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:wire_trace_reader",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_test(
    name = "data_provider_test",
    srcs = ["data_provider_test.cc"],
    deps = [
        ":constants",
        ":data_provider",
        ":time_range",
        ":timeline",
        "//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "//third_party/tcmalloc:malloc_extension",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@imgui",
        "@org_xprof//frontend/app/components/trace_viewer_v2:event_data",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_test(
    name = "data_provider_parity_test",
    srcs = ["data_provider_parity_test.cc"],
    deps = [
        ":data_provider",
        ":timeline",
        "//plugin/xprof/protobuf:trace_data_response_proto_cc",
        "//plugin/xprof/protobuf:trace_events_proto_cc",
        "//plugin/xprof/protobuf:trace_events_raw_proto_cc",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_parser_core",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_stream_ingestor",
        "@org_xprof//xprof/convert/trace_viewer:trace_events",
        "@org_xprof//xprof/convert/trace_viewer:trace_events_to_json",
        "@org_xprof//xprof/convert/trace_viewer:trace_options",
        "@org_xprof//xprof/convert/trace_viewer/delta_series:trace_data_to_compressed_delta_series_proto",
        "@org_xprof//xprof/convert/trace_viewer/delta_series:zstd_compression",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)
