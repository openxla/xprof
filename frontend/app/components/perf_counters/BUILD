load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/chart/dashboard",
        "//frontend/app/components/chart/table",
        "//frontend/app/components/controls/category_filter",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/controls/string_filter",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "perf_counters_css",
    src = "perf_counters.scss",
    # stack = True,
    sourcemap = False,
)
