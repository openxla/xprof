load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//@types/google.visualization",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/overview_page/inference_latency_chart",
        "@org_xprof//frontend/app/components/overview_page/normalized_accelerator_performance_view",
        "@org_xprof//frontend/app/components/overview_page/performance_summary",
        "@org_xprof//frontend/app/components/overview_page/run_environment_view",
        "@org_xprof//frontend/app/components/overview_page/step_time_graph",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "overview_page_css",
    src = "overview_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
