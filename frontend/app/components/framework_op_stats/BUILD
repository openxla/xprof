load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "framework_op_stats_adapter",
    srcs = [
        "framework_op_stats_adapter.ts",
    ],
    deps = [
        ":framework_op_stats",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/constants",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
        "//frontend/app/store/framework_op_stats",
    ],
)

xprof_ng_module(
    name = "framework_op_stats",
    srcs = [
        "framework_op_stats.ts",
        "framework_op_stats_module.ts",
    ],
    assets = [
        ":framework_op_stats_css",
        "framework_op_stats.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_menu",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:chart_options",
        "//frontend/app/components/chart:data_processor",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/framework_op_stats/flop_rate_chart",
        "//frontend/app/components/framework_op_stats/model_properties",
        "//frontend/app/components/framework_op_stats/operations_table",
        "//frontend/app/components/framework_op_stats/stats_table",
        "//frontend/app/store/framework_op_stats",
    ],
)

sass_binary(
    name = "framework_op_stats_css",
    src = "framework_op_stats.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
