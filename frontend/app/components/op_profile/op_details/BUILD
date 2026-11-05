load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "op_details",
    srcs = [
        "op_details.ts",
        "op_details_module.ts",
    ],
    assets = [
        ":op_details_css",
        "op_details.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/ngx-json-viewer",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:op_metrics_proto_defs",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/common/utils",
        "//frontend/app/services/communication_service",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "op_details_css",
    src = "op_details.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
