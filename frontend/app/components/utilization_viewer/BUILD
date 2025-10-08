load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = [
    "//perftools/accelerators/xprof/frontend:__subpackages__",
    "@org_xprof//frontend:__subpackages__",
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
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:data_processor",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
        "@org_xprof//frontend/app/components/chart/dashboard",
        "@org_xprof//frontend/app/components/controls/category_filter",
        "@org_xprof//frontend/app/components/controls/export_as_csv",
        "@org_xprof//frontend/app/components/controls/view_architecture",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "utilization_viewer_css",
    src = "utilization_viewer.scss",
    # stack = True,
    sourcemap = False,
)
