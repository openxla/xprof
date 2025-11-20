# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

cc_library(
    name = "time_formatter",
    srcs = ["time_formatter.cc"],
    hdrs = ["time_formatter.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
    ],
)

cc_test(
    name = "time_formatter_test",
    srcs = ["time_formatter_test.cc"],
    deps = [
        ":time_formatter",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
