load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "max_heap_chart",
    srcs = [
        "max_heap_chart.ts",
        "max_heap_chart_module.ts",
    ],
    allow_warnings = True,
    assets = [
        ":max_heap_chart_css",
        "max_heap_chart.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "max_heap_chart_css",
    src = "max_heap_chart.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
