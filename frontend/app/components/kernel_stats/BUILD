load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "kernel_stats_adapter",
    srcs = [
        "kernel_stats_adapter.ts",
    ],
    deps = [
        ":kernel_stats",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "kernel_stats",
    srcs = [
        "kernel_stats.ts",
        "kernel_stats_module.ts",
    ],
    assets = [
        ":kernel_stats_css",
        "kernel_stats.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/controls/export_as_csv",
        "@org_xprof//frontend/app/components/kernel_stats/kernel_stats_table",
        "@org_xprof//frontend/app/store/common_data_store",
    ],
)

sass_binary(
    name = "kernel_stats_css",
    src = "kernel_stats.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
