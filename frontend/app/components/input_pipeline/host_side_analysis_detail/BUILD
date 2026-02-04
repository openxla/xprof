load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "host_side_analysis_detail",
    srcs = [
        "host_side_analysis_detail.ts",
        "host_side_analysis_detail_module.ts",
    ],
    assets = [
        ":host_side_analysis_detail_css",
        "host_side_analysis_detail.ng.html",
    ],
    # strict_templates = False,
    deps = [
        ":host_side_analysis_detail_table_data_provider",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

xprof_ng_module(
    name = "host_side_analysis_detail_table_data_provider",
    srcs = [
        "host_side_analysis_detail_table_data_provider.ts",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

sass_binary(
    name = "host_side_analysis_detail_css",
    src = "host_side_analysis_detail.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
