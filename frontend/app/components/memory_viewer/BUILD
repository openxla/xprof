load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_viewer",
    srcs = [
        "memory_viewer.ts",
        "memory_viewer_module.ts",
    ],
    assets = [
        ":memory_viewer_css",
        "memory_viewer.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/memory_viewer/memory_usage",
        "@org_xprof//frontend/app/components/memory_viewer/memory_viewer_control",
        "@org_xprof//frontend/app/components/memory_viewer/memory_viewer_main",
        "@org_xprof//frontend/app/services/data_service",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "memory_viewer_css",
    src = "memory_viewer.scss",
    # stack = False,
    sourcemap = False,
    # strict_deps = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
