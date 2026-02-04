load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "flop_rate_chart",
    srcs = [
        "flop_rate_chart.ts",
        "flop_rate_chart_module.ts",
    ],
    assets = [
        ":flop_rate_chart_css",
        "flop_rate_chart.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:data_processor",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

sass_binary(
    name = "flop_rate_chart_css",
    src = "flop_rate_chart.scss",
    # stack = False,
    sourcemap = False,
)
