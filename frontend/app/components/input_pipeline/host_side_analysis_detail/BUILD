load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
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
        "@npm//@angular/core",
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
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "host_side_analysis_detail_test.ts",
    ],
    deps = [
        ":host_side_analysis_detail",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
        "@org_xprof//frontend/app/components/chart:chart_options",
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
