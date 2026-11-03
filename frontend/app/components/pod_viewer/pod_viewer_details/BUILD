load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "pod_viewer_details",
    srcs = [
        "pod_viewer_details.ts",
        "pod_viewer_details_module.ts",
    ],
    assets = [
        ":pod_viewer_details_css",
        "pod_viewer_details.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "pod_viewer_details_css",
    src = "pod_viewer_details.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
