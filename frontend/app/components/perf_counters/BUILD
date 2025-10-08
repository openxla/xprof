load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//javascript/angular2:build_defs.bzl", "ng_module")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "perf_counters",
    srcs = [
        "perf_counters.ts",
        "perf_counters_module.ts",
    ],
    assets = [
        ":perf_counters_css",
        "perf_counters.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//@types/google.visualization",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/chart/dashboard",
        "@org_xprof//frontend/app/components/chart/table",
        "@org_xprof//frontend/app/components/controls/category_filter",
        "@org_xprof//frontend/app/components/controls/export_as_csv",
        "@org_xprof//frontend/app/components/controls/string_filter",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "perf_counters_css",
    src = "perf_counters.scss",
    # stack = True,
    sourcemap = False,
)
