# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

wasm_cc_library(
    name = "colors",
    srcs = ["color_generator.cc"],
    hdrs = [
        "color_generator.h",
        "colors.h",
    ],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@imgui",
    ],
)

cc_test(
    name = "color_generator_test",
    srcs = ["color_generator_test.cc"],
    deps = [
        ":colors",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@imgui",
    ],
)
