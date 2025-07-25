load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//javascript/angular2:build_defs.bzl", "ng_module")

package(default_visibility = ["//frontend:internal"])

ng_module(
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
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/interfaces:smart_suggestion",
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
