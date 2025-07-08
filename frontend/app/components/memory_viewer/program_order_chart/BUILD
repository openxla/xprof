load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "program_order_chart",
    srcs = [
        "program_order_chart.ts",
        "program_order_chart_module.ts",
    ],
    assets = [
        ":program_order_chart_css",
        "program_order_chart.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

sass_binary(
    name = "program_order_chart_css",
    src = "program_order_chart.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
