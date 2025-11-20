# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

cc_library(
    name = "trace_event",
    hdrs = ["trace_event.h"],
)

wasm_cc_library(
    name = "trace_event_parser",
    srcs = ["trace_event_parser.cc"],
    hdrs = ["trace_event_parser.h"],
    deps = [
        ":trace_event",
        "//third_party/emscripten:embind",
        "@org_xprof//frontend/app/components/trace_viewer_v2:application",
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline:data_provider",
    ],
    alwayslink = True,
)

cc_library(
    name = "trace_event_tree",
    srcs = ["trace_event_tree.cc"],
    hdrs = ["trace_event_tree.h"],
    deps = [
        ":trace_event",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "trace_event_tree_test",
    srcs = ["trace_event_tree_test.cc"],
    deps = [
        ":trace_event",
        ":trace_event_tree",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
