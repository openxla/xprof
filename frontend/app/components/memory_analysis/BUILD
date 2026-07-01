load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_analysis",
    srcs = [
        "memory_analysis.ts",
        "memory_analysis_module.ts",
    ],
    assets = [
        ":memory_analysis_css",
        "memory_analysis.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/angular:angular_material_chips",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_sort",
        "@org_xprof//frontend/app/common/angular:angular_material_table",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "memory_analysis_css",
    src = "memory_analysis.scss",
    # stack = True,
    sourcemap = False,
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "memory_analysis_test.ts",
    ],
    deps = [
        ":memory_analysis",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/common",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)
