load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "framework_op_stats_adapter",
    srcs = [
        "framework_op_stats_adapter.ts",
    ],
    deps = [
        ":framework_op_stats",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
        "@org_xprof//frontend/app/store/framework_op_stats",
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
        "@npm//@angular/core",
        "@npm//@ngrx/store",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_menu",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:data_processor",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
        "@org_xprof//frontend/app/components/controls/export_as_csv",
        "@org_xprof//frontend/app/components/framework_op_stats/flop_rate_chart",
        "@org_xprof//frontend/app/components/framework_op_stats/model_properties",
        "@org_xprof//frontend/app/components/framework_op_stats/operations_table",
        "@org_xprof//frontend/app/components/framework_op_stats/stats_table",
        "@org_xprof//frontend/app/store/framework_op_stats",
    ],
)

sass_binary(
    name = "framework_op_stats_css",
    src = "framework_op_stats.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
