load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "operation_level_analysis",
    srcs = [
        "operation_level_analysis.ts",
        "operation_level_analysis_module.ts",
    ],
    assets = [
        ":operation_level_analysis_css",
        "operation_level_analysis.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_slide_toggle",
        "//frontend/app/common/constants:roofline_model_constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:chart_options",
        "//frontend/app/components/chart:data_processor",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart/dashboard",
        "//frontend/app/components/chart/table",
        "//frontend/app/components/controls/category_filter",
        "//frontend/app/components/controls/string_filter",
        "//frontend/app/components/stack_trace_snippet",
    ],
)

sass_binary(
    name = "operation_level_analysis_css",
    src = "operation_level_analysis.scss",
    # stack = True,
    sourcemap = False,
)
