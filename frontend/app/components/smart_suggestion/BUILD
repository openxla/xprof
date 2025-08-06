load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "smart_suggestion",
    srcs = [
        "smart_suggestion_view.ts",
    ],
    assets = [
        ":smart_suggestion_view_css",
        "smart_suggestion_view.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "smart_suggestion_view_css",
    src = "smart_suggestion_view.scss",
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
