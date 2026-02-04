load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "org_chart",
    srcs = [
        "org_chart.ts",
        "org_chart_module.ts",
    ],
    assets = [
        ":org_chart_css",
        "org_chart.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
    ],
)

sass_binary(
    name = "org_chart_css",
    src = "org_chart.scss",
    # stack = False,
    sourcemap = False,
)
