load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@org_xprof//frontend/app/components/trace_viewer_v2:build_defs.bzl", "wasm_cc_library")
load(":font_rules.bzl", "imgui_font_headers")

package(default_visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"])

exports_files(
    ["trace_viewer_icons.ttf"],
    visibility = ["//frontend:internal"],
)

bzl_library(
    name = "font_rules_bzl",
    srcs = ["font_rules.bzl"],
    visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"],
    deps = [
        "//third_party/bazel_rules/rules_cc/cc/common",
    ],
)

genrule(
    name = "roboto_ttf",
    srcs = ["@roboto_font//file"],
    outs = ["Roboto[wdth,wght].ttf"],
    cmd = "cp $< $@",
)

imgui_font_headers(
    name = "font_headers_fileset",
    srcs = [":roboto_ttf"],
    # Embed the raw (uncompressed) TTF bytes so that the font atlas can
    # reference them in place without decompressing a heap copy at runtime.
    args = [
        "-u8",
        "-nocompress",
    ],
)

imgui_font_headers(
    name = "trace_viewer_icons_header",
    srcs = ["trace_viewer_icons.ttf"],
    args = [
        "-u8",
        "-nocompress",
    ],
)

wasm_cc_library(
    name = "font_headers",
    hdrs = [
        ":font_headers_fileset",
        ":trace_viewer_icons_header",
    ],
)

wasm_cc_library(
    name = "fonts",
    srcs = ["fonts.cc"],
    hdrs = ["fonts.h"],
    defines = [
        "IMGUI_DEFINE_MATH_OPERATORS",
    ],
    deps = [
        ":font_headers",
        "@com_google_absl//absl/log",
        "@imgui",
    ],
)
