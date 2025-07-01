load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "roofline_model",
    srcs = [
        "roofline_model.ts",
        "roofline_model_module.ts",
    ],
    assets = [
        ":roofline_model_css",
        "roofline_model.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/constants:roofline_model_constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/chart/table",
        "@org_xprof//frontend/app/components/controls/category_filter",
        "@org_xprof//frontend/app/components/controls/export_as_csv",
        "@org_xprof//frontend/app/components/controls/string_filter",
        "@org_xprof//frontend/app/components/roofline_model/operation_level_analysis",
        "@org_xprof//frontend/app/components/roofline_model/program_level_analysis",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "roofline_model_css",
    src = "roofline_model.scss",
    # stack = True,
    sourcemap = False,
)
