load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "hlo_text_view",
    srcs = [
        "hlo_text_view.ts",
        "hlo_text_view_module.ts",
    ],
    assets = [
        ":hlo_text_view_css",
        "hlo_text_view.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/classes",
        "//frontend/app/common/constants",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "hlo_text_view_css",
    src = "hlo_text_view.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
