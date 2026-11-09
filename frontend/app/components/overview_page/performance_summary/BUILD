load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "performance_summary",
    srcs = [
        "performance_summary.ts",
        "performance_summary_module.ts",
    ],
    assets = [
        ":performance_summary_css",
        "performance_summary.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "performance_summary_css",
    src = "performance_summary.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
