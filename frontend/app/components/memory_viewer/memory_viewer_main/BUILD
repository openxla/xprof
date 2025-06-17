load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//angular-split",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_slide_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/memory_viewer/max_heap_chart",
        "@org_xprof//frontend/app/components/memory_viewer/memory_usage",
        "@org_xprof//frontend/app/components/memory_viewer/program_order_chart",
        "@org_xprof//frontend/app/components/stack_trace_snippet",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "memory_viewer_main_css",
    src = "memory_viewer_main.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
