load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_viewer_main",
    srcs = [
        "memory_viewer_main.ts",
        "memory_viewer_main_module.ts",
    ],
    assets = [
        ":memory_viewer_main_css",
        "memory_viewer_main.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/angular-split",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_slide_toggle",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/memory_viewer/buffer_allocation_timeline",
        "//frontend/app/components/memory_viewer/max_heap_chart",
        "//frontend/app/components/memory_viewer/memory_usage",
        "//frontend/app/components/memory_viewer/program_order_chart",
        "//frontend/app/components/source_mapper",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "memory_viewer_main_css",
    src = "memory_viewer_main.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
