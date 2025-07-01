load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "stack_bar_chart",
    srcs = [
        "stack_bar_chart.ts",
        "stack_bar_chart_module.ts",
    ],
    assets = [
        ":stack_bar_chart_css",
        "stack_bar_chart.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "stack_bar_chart_css",
    src = "stack_bar_chart.scss",
    # stack = False,
    sourcemap = False,
)
