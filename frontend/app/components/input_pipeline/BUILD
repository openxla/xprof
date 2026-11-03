load("//defs:defs.bzl", "sass_binary", "sass_library", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/classes",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/input_pipeline/analysis_summary",
        "//frontend/app/components/input_pipeline/device_side_analysis_detail",
        "//frontend/app/components/input_pipeline/host_op",
        "//frontend/app/components/input_pipeline/host_side_analysis_detail",
        "//frontend/app/components/input_pipeline/max_infeed_detail",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "input_pipeline_css",
    src = "input_pipeline.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)

sass_library(
    name = "common_css",
    srcs = ["input_pipeline.scss"],
    deps = [
        "//frontend/app/styles:common",
    ],
)
