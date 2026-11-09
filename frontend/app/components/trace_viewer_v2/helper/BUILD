load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "//frontend/app/components/trace_viewer_v2:__subpackages__",
])

cc_library(
    name = "time_formatter",
    srcs = ["time_formatter.cc"],
    hdrs = ["time_formatter.h"],
    deps = [
        "//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

wasm_cc_library(
    name = "clipboard",
    srcs = ["clipboard.cc"],
    hdrs = ["clipboard.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "time_formatter_test",
    srcs = ["time_formatter_test.cc"],
    deps = [
        ":time_formatter",
        "//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "counter_formatter",
    srcs = ["counter_formatter.cc"],
    hdrs = ["counter_formatter.h"],
    deps = [
        ":time_formatter",
        "//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "counter_formatter_test",
    srcs = ["counter_formatter_test.cc"],
    deps = [
        ":counter_formatter",
        "//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
