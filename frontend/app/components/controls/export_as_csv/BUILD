load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "export_as_csv",
    srcs = [
        "export_as_csv.ts",
        "export_as_csv_module.ts",
    ],
    assets = [
        "export_as_csv.ng.html",
        ":export_as_csv_css",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "export_as_csv_css",
    src = "export_as_csv.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
