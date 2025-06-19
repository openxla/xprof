load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "pod_viewer",
    srcs = [
        "pod_viewer.ts",
        "pod_viewer_common.ts",
        "pod_viewer_module.ts",
    ],
    assets = [
        ":pod_viewer_css",
        "pod_viewer.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/angular:angular_material_slider",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/pod_viewer/stack_bar_chart",
        "@org_xprof//frontend/app/components/pod_viewer/topology_graph",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "pod_viewer_css",
    src = "pod_viewer.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
