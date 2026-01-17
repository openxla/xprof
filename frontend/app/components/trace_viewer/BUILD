load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "trace_viewer",
    srcs = [
        "trace_viewer.ts",
        "trace_viewer_module.ts",
    ],
    assets = [
        ":trace_viewer_css",
        "trace_viewer.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/pipes",
        "@org_xprof//frontend/app/services/communication_service",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "trace_viewer_css",
    src = "trace_viewer.scss",
    # stack = False,
    sourcemap = False,
)
