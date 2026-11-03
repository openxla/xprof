load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "max_heap_chart_css",
    src = "max_heap_chart.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
