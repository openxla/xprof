load("//defs:defs.bzl", "xprof_ng_module", ts_library = "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

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
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
    ],
)

ts_library(
    name = "device_side_analysis_detail_data_provider",
    srcs = [
        "device_side_analysis_detail_data_provider.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

sass_binary(
    name = "device_side_analysis_detail_css",
    src = "device_side_analysis_detail.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
