load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "kernel_stats_table_css",
    src = "kernel_stats_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
