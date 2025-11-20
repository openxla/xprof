# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = ["//third_party/dear_imgui"],
)

cc_library(
    name = "time_range",
    srcs = ["time_range.cc"],
    hdrs = ["time_range.h"],
    deps = [
        "@com_google_absl//absl/log",
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

cc_library(
    name = "timeline",
    srcs = [
        "timeline.cc",
    ],
    hdrs = [
        "timeline.h",
    ],
    deps = [
        ":constants",
        ":time_range",
        "//third_party/dear_imgui",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_xprof//frontend/app/components/trace_viewer_v2:animation",
        "@org_xprof//frontend/app/components/trace_viewer_v2:event_data",
        "@org_xprof//frontend/app/components/trace_viewer_v2/color:colors",
        "@org_xprof//frontend/app/components/trace_viewer_v2/helper:time_formatter",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
    ],
)

cc_test(
    name = "timeline_test",
    srcs = ["timeline_test.cc"],
    deps = [
        ":constants",
        ":time_range",
        ":timeline",
        "//third_party/dear_imgui",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//frontend/app/components/trace_viewer_v2:animation",
        "@org_xprof//frontend/app/components/trace_viewer_v2:event_data",
    ],
)

cc_library(
    name = "data_provider",
    srcs = [
        "data_provider.cc",
    ],
    hdrs = ["data_provider.h"],
    deps = [
        ":time_range",
        ":timeline",
        "//util/gtl:comparator",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_tree",
    ],
)

cc_test(
    name = "data_provider_test",
    srcs = ["data_provider_test.cc"],
    deps = [
        ":data_provider",
        ":time_range",
        ":timeline",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event",
    ],
)
