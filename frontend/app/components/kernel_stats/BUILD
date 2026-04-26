load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "kernel_stats_adapter",
    srcs = [
        "kernel_stats_adapter.ts",
    ],
    deps = [
        ":kernel_stats",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "kernel_stats",
    srcs = [
        "kernel_stats.ts",
        "kernel_stats_module.ts",
    ],
    assets = [
        ":kernel_stats_css",
        "kernel_stats.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/controls/export_as_csv",
        "@org_xprof//frontend/app/components/kernel_stats/kernel_stats_table",
        "@org_xprof//frontend/app/store/common_data_store",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "kernel_stats_adapter_test.ts",
        "kernel_stats_test.ts",
    ],
    deps = [
        ":kernel_stats",
        ":kernel_stats_adapter",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/utils:testing",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "kernel_stats_css",
    src = "kernel_stats.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
