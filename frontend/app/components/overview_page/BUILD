load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "overview_page",
    srcs = [
        "overview_page_module.ts",
    ],
    assets = [
        ":overview_page_css",
        "overview_page.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//:node_modules/rxjs",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/overview_page/inference_latency_chart",
        "//frontend/app/components/overview_page/normalized_accelerator_performance_view",
        "//frontend/app/components/overview_page/performance_summary",
        "//frontend/app/components/overview_page/run_environment_view",
        "//frontend/app/components/overview_page/step_time_graph",
        "//frontend/app/components/smart_suggestion",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "overview_page_css",
    src = "overview_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
