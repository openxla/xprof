load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")
load(":font_rules.bzl", "imgui_font_headers")

package(default_visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"])

bzl_library(
    name = "font_rules_bzl",
    srcs = ["font_rules.bzl"],
    visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"],
    deps = [
        "//third_party/bazel_rules/rules_cc/cc/common",
    ],
)

wasm_cc_library(
    name = "fonts",
    srcs = ["fonts.cc"],
    hdrs = ["fonts.h"],
    defines = [
        "IMGUI_DEFINE_MATH_OPERATORS",
        "IMGUI_ENABLE_FREETYPE",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@imgui",
        "@imgui//:imgui_freetype",
    ],
)
