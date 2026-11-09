load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = [
    "//frontend:internal",
])

xprof_ng_module(
    name = "utilization_viewer",
    srcs = [
        "utilization_viewer.ts",
        "utilization_viewer_module.ts",
    ],
    assets = [
        ":utilization_viewer_css",
        "utilization_viewer.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/common/utils",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:chart_options",
        "//frontend/app/components/chart:data_processor",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart/dashboard",
        "//frontend/app/components/controls/category_filter",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/controls/view_architecture",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "utilization_viewer_css",
    src = "utilization_viewer.scss",
    # stack = True,
    sourcemap = False,
)
