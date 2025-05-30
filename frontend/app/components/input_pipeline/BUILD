load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "sass_library")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "input_pipeline",
    srcs = [
        "input_pipeline.ts",
        "input_pipeline_common.ts",
        "input_pipeline_module.ts",
    ],
    assets = [
        ":input_pipeline_css",
        "input_pipeline.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/input_pipeline/analysis_summary",
        "@org_xprof//frontend/app/components/input_pipeline/device_side_analysis_detail",
        "@org_xprof//frontend/app/components/input_pipeline/host_op",
        "@org_xprof//frontend/app/components/input_pipeline/host_side_analysis_detail",
        "@org_xprof//frontend/app/components/input_pipeline/max_infeed_detail",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "input_pipeline_css",
    src = "input_pipeline.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

sass_library(
    name = "common_css",
    srcs = ["input_pipeline.scss"],
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
