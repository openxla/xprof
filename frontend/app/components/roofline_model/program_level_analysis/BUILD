load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "program_level_analysis",
    srcs = [
        "program_level_analysis.ts",
        "program_level_analysis_module.ts",
    ],
    assets = [
        ":program_level_analysis_css",
        "program_level_analysis.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:chart_options",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart/dashboard",
        "//frontend/app/components/chart/table",
        "//frontend/app/components/controls/category_filter",
    ],
)

sass_binary(
    name = "program_level_analysis_css",
    src = "program_level_analysis.scss",
    # stack = True,
    sourcemap = False,
)
