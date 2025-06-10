load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "sidenav",
    srcs = [
        "sidenav.ts",
        "sidenav_module.ts",
    ],
    assets = [
        ":sidenav_css",
        "sidenav.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/capture_profile",
        "@org_xprof//frontend/app/components/memory_viewer/buffer_details",
        "@org_xprof//frontend/app/components/op_profile/op_details",
        "@org_xprof//frontend/app/components/pod_viewer/pod_viewer_details",
        "@org_xprof//frontend/app/services/communication_service",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "sidenav_css",
    src = "sidenav.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
