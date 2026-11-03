load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "analysis_summary",
    srcs = [
        "analysis_summary.ts",
        "analysis_summary_module.ts",
    ],
    assets = [
        ":analysis_summary_css",
        "analysis_summary.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "analysis_summary_css",
    src = "analysis_summary.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
