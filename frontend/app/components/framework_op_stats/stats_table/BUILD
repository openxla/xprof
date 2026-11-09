load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "stats_table",
    srcs = [
        "stats_table.ts",
        "stats_table_module.ts",
    ],
    assets = [
        ":stats_table_css",
        "stats_table.ng.html",
    ],
    # strict_templates = False,
    deps = [
        ":stats_table_data_provider",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
    ],
)

xprof_ng_module(
    name = "stats_table_data_provider",
    srcs = [
        "stats_table_data_provider.ts",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart:utils",
    ],
)

sass_binary(
    name = "stats_table_css",
    src = "stats_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
