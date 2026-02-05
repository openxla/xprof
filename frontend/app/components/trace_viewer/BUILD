load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "trace_viewer",
    srcs = [
        "trace_viewer.ts",
        "trace_viewer_container.ts",
        "trace_viewer_module.ts",
    ],
    assets = [
        ":trace_viewer_css",
        ":trace_viewer_container_css",
        "trace_viewer.ng.html",
        "trace_viewer_container.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "@org_xprof//frontend/app/pipes",
        "@org_xprof//frontend/app/services/communication_service",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "trace_viewer_css",
    src = "trace_viewer.scss",
    # stack = False,
    sourcemap = False,
)

sass_binary(
    name = "trace_viewer_container_css",
    src = "trace_viewer_container.scss",
    # stack = False,
    sourcemap = False,
)
