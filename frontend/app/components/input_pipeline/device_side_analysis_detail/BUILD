load("//defs:defs.bzl", "sass_binary", "ts_library", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "device_side_analysis_detail",
    srcs = [
        "device_side_analysis_detail.ts",
        "device_side_analysis_detail_module.ts",
    ],
    assets = [
        ":device_side_analysis_detail_css",
        "device_side_analysis_detail.ng.html",
    ],
    # strict_templates = False,
    deps = [
        ":device_side_analysis_detail_data_provider",
        "//:node_modules/@angular/core",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
    ],
)

ts_library(
    name = "device_side_analysis_detail_data_provider",
    srcs = [
        "device_side_analysis_detail_data_provider.ts",
    ],
    deps = [
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils:chart_utils",
        "//frontend/app/components/chart:default_data_provider",
    ],
)

sass_binary(
    name = "device_side_analysis_detail_css",
    src = "device_side_analysis_detail.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
