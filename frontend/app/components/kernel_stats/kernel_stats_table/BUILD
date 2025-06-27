load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "kernel_stats_table",
    srcs = [
        "kernel_stats_table.ts",
        "kernel_stats_table_module.ts",
    ],
    assets = [
        ":kernel_stats_table_css",
        "kernel_stats_table.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "kernel_stats_table_css",
    src = "kernel_stats_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
