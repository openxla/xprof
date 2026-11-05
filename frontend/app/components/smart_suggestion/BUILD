load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "smart_suggestion",
    srcs = [
        "smart_suggestion_view.ts",
    ],
    assets = [
        ":smart_suggestion_view_css",
        "smart_suggestion_view.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_progress_spinner",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "smart_suggestion_view_css",
    src = "smart_suggestion_view.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
