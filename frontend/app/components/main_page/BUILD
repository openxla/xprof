load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/angular:angular_material_toolbar",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/empty_page",
        "//frontend/app/components/framework_op_stats:framework_op_stats_adapter",
        "//frontend/app/components/graph_viewer",
        "//frontend/app/components/hlo_stats",
        "//frontend/app/components/inference_profile",
        "//frontend/app/components/input_pipeline",
        "//frontend/app/components/kernel_stats:kernel_stats_adapter",
        "//frontend/app/components/megascale_perfetto",
        "//frontend/app/components/megascale_stats",
        "//frontend/app/components/memory_profile",
        "//frontend/app/components/memory_viewer",
        "//frontend/app/components/op_profile",
        "//frontend/app/components/overview_page",
        "//frontend/app/components/perf_counters",
        "//frontend/app/components/pod_viewer",
        "//frontend/app/components/rack/navigation",
        "//frontend/app/components/roofline_model",
        "//frontend/app/components/sidenav",
        "//frontend/app/components/stack_trace_page",
        "//frontend/app/components/static_kernel_viewer",
        "//frontend/app/components/trace_viewer",
        "//frontend/app/components/utilization_viewer",
        "//frontend/app/services/communication_service",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "main_page_css",
    src = "main_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
