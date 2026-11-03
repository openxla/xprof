load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_slide_toggle",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/classes",
        "//frontend/app/common/constants:roofline_model_constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/chart/table",
        "//frontend/app/components/controls/category_filter",
        "//frontend/app/components/controls/export_as_csv",
        "//frontend/app/components/controls/string_filter",
        "//frontend/app/components/roofline_model/operation_level_analysis",
        "//frontend/app/components/roofline_model/program_level_analysis",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "roofline_model_css",
    src = "roofline_model.scss",
    # stack = True,
    sourcemap = False,
    deps = ["//frontend/app/styles:common"],
)
