load("@npm//@bazel/concatjs:index.bzl", "ts_library")

# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

ts_library(
    name = "event_args_keys",
    srcs = ["event_args_keys.ts"],
)

cc_library(
    name = "trace_event",
    hdrs = ["trace_event.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:context_types",
    ],
)

wasm_cc_library(
    name = "trace_event_parser",
    srcs = ["trace_event_parser.cc"],
    hdrs = ["trace_event_parser.h"],
    deps = [
        ":trace_event",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@org_xprof//frontend/app/components/trace_viewer_v2:application",
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline:data_provider",
        "@tsl//tsl/platform:fingerprint",
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
