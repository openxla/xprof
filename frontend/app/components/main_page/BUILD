load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "main_page",
    srcs = [
        "main_page.ts",
        "main_page_module.ts",
    ],
    assets = [
        ":main_page_css",
        "main_page.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_sidenav",
        "@org_xprof//frontend/app/common/angular:angular_material_toolbar",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/empty_page",
        "@org_xprof//frontend/app/components/framework_op_stats:framework_op_stats_adapter",
        "@org_xprof//frontend/app/components/graph_viewer",
        "@org_xprof//frontend/app/components/hlo_stats",
        "@org_xprof//frontend/app/components/inference_profile",
        "@org_xprof//frontend/app/components/input_pipeline",
        "@org_xprof//frontend/app/components/kernel_stats:kernel_stats_adapter",
        "@org_xprof//frontend/app/components/megascale_stats",
        "@org_xprof//frontend/app/components/memory_profile",
        "@org_xprof//frontend/app/components/memory_viewer",
        "@org_xprof//frontend/app/components/op_profile",
        "@org_xprof//frontend/app/components/overview_page",
        "@org_xprof//frontend/app/components/pod_viewer",
        "@org_xprof//frontend/app/components/roofline_model",
        "@org_xprof//frontend/app/components/sidenav",
        "@org_xprof//frontend/app/components/stack_trace_page",
        "@org_xprof//frontend/app/components/trace_viewer",
        "@org_xprof//frontend/app/services/communication_service",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "main_page_css",
    src = "main_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
