load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
        "@org_xprof//frontend/app/components/chart/dashboard",
        "@org_xprof//frontend/app/components/chart/table",
        "@org_xprof//frontend/app/components/controls/category_filter",
    ],
)

sass_binary(
    name = "program_level_analysis_css",
    src = "program_level_analysis.scss",
    # stack = True,
    sourcemap = False,
)
