load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_slide_toggle",
        "@org_xprof//frontend/app/common/constants:roofline_model_constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:data_processor",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
        "@org_xprof//frontend/app/components/chart/dashboard",
        "@org_xprof//frontend/app/components/chart/table",
        "@org_xprof//frontend/app/components/controls/category_filter",
        "@org_xprof//frontend/app/components/controls/string_filter",
        "@org_xprof//frontend/app/components/stack_trace_snippet",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "operation_level_analysis_css",
    src = "operation_level_analysis.scss",
    # stack = True,
    sourcemap = False,
)
