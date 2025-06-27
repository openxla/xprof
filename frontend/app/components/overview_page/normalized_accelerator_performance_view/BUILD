load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "normalized_accelerator_performance_view",
    srcs = [
        "normalized_accelerator_performance_view.ts",
        "normalized_accelerator_performance_view_module.ts",
    ],
    assets = [
        ":normalized_accelerator_performance_view_css",
        "normalized_accelerator_performance_view.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "normalized_accelerator_performance_view_css",
    src = "normalized_accelerator_performance_view.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
