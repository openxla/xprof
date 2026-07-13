load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "string_filter",
    srcs = [
        "string_filter.ts",
        "string_filter_module.ts",
    ],
    assets = [
        "string_filter.ng.html",
        "string_filter_css",
    ],
    # strict_templates = False,
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
    ],
)

sass_binary(
    name = "string_filter_css",
    src = "string_filter.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
