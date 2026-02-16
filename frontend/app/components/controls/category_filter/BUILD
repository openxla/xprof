load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "category_filter",
    srcs = [
        "category_filter.ts",
        "category_filter_module.ts",
    ],
    assets = [
        "category_filter.ng.html",
        ":category_filter_css",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
    ],
)

sass_binary(
    name = "category_filter_css",
    src = "category_filter.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
