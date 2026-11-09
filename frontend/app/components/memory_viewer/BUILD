load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/memory_viewer/buffer_details",
        "//frontend/app/components/memory_viewer/max_heap_chart_downloader",
        "//frontend/app/components/memory_viewer/memory_usage",
        "//frontend/app/components/memory_viewer/memory_viewer_control",
        "//frontend/app/components/memory_viewer/memory_viewer_main",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "memory_viewer_css",
    src = "memory_viewer.scss",
    # stack = False,
    sourcemap = False,
    # strict_deps = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
