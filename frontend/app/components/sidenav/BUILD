load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/capture_kernel",
        "//frontend/app/components/capture_profile",
        "//frontend/app/components/memory_viewer/buffer_details",
        "//frontend/app/components/op_profile/op_details",
        "//frontend/app/components/pod_viewer/pod_viewer_details",
        "//frontend/app/services/communication_service",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "sidenav_css",
    src = "sidenav.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
