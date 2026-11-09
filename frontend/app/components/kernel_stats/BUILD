load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/store",
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
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/kernel_stats/kernel_stats_table",
        "//frontend/app/store/common_data_store",
    ],
)

sass_binary(
    name = "kernel_stats_css",
    src = "kernel_stats.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
