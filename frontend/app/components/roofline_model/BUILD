load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//@types/google.visualization",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_slide_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/classes",
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
    deps = ["@org_xprof//frontend/app/styles:common"],
)
