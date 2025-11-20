load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")
load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")
load(":font_rules.bzl", "imgui_font_headers")

package(default_visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"])

bzl_library(
    name = "font_rules_bzl",
    srcs = ["font_rules.bzl"],
    parse_tests = False,
    visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"],
    deps = [
        "//third_party/bazel_rules/rules_cc/cc/common",
    ],
)

imgui_font_headers(
    name = "font_headers_fileset",
    srcs = ["//third_party/roboto:roboto_unhinted"],
    args = ["-base85"],
)

wasm_cc_library(
    name = "font_headers",
    hdrs = [":font_headers_fileset"],
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
        ":font_headers",
        "//third_party/dear_imgui",
        "//third_party/dear_imgui:imgui_freetype",
        "@com_google_absl//absl/log",
    ],
)
