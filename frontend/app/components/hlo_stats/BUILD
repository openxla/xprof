load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "hlo_stats",
    srcs = [
        "hlo_stats.ts",
        "hlo_stats_module.ts",
    ],
    assets = [
        ":hlo_stats_css",
        "hlo_stats.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/angular:angular_material_slide_toggle",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/classes",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/common/utils",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:chart_options",
        "//frontend/app/components/chart:data_processor",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart/dashboard",
        "//frontend/app/components/controls/category_filter",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/controls/string_filter",
        "//frontend/app/components/framework_op_stats/flop_rate_chart",
        "//frontend/app/components/stack_trace_snippet",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "hlo_stats_css",
    src = "hlo_stats.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
