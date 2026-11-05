load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "megascale_stats",
    srcs = [
        "megascale_stats.ts",
        "megascale_stats_module.ts",
    ],
    assets = [
        ":megascale_stats_css",
        "megascale_stats.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/platform-browser",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//:node_modules/rxjs",
        "//:node_modules/safevalues",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/common/utils",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:chart_options",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart/dashboard",
        "//frontend/app/components/controls/category_filter",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "megascale_stats_css",
    src = "megascale_stats.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
