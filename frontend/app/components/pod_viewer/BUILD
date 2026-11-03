load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_slider",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/pod_viewer/stack_bar_chart",
        "//frontend/app/components/pod_viewer/topology_graph",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "pod_viewer_css",
    src = "pod_viewer.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
