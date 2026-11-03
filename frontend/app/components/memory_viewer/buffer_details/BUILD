load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "buffer_details",
    srcs = [
        "buffer_details.ts",
        "buffer_details_module.ts",
    ],
    assets = [
        ":buffer_details_css",
        "buffer_details.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "buffer_details_css",
    src = "buffer_details.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
