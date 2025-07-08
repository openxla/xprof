load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

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
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "analysis_summary_css",
    src = "analysis_summary.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
