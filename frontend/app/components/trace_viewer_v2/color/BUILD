load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "//frontend/app/components/trace_viewer_v2:__subpackages__",
])

wasm_cc_library(
    name = "colors",
    srcs = [
        "color_generator.cc",
        "colors.cc",
    ],
    hdrs = [
        "color_generator.h",
        "colors.h",
        "palettes.h",
    ],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@imgui",
        "@tsl//tsl/platform:fingerprint",
    ],
)

wasm_cc_library(
    name = "counter_colors",
    srcs = ["counter_colors.cc"],
    hdrs = ["counter_colors.h"],
    deps = [
        ":colors",
        "@com_google_absl//absl/strings",
        "@imgui",
    ],
)

cc_test(
    name = "counter_colors_test",
    srcs = ["counter_colors_test.cc"],
    deps = [
        ":colors",
        ":counter_colors",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@imgui",
    ],
)

cc_test(
    name = "color_generator_test",
    srcs = ["color_generator_test.cc"],
    deps = [
        ":colors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@imgui",
    ],
)
