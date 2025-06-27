load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "diagnostics_view",
    srcs = [
        "diagnostics_view.ts",
        "diagnostics_view_module.ts",
    ],
    assets = [
        ":diagnostics_view_css",
        "diagnostics_view.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "diagnostics_view_css",
    src = "diagnostics_view.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
