load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "download_hlo",
    srcs = [
        "blob_downloader.ts",
        "download_hlo.ts",
        "download_hlo_module.ts",
    ],
    assets = [
        ":download_hlo_css",
        "download_hlo.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//rxjs",
        "@npm//safevalues",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_menu",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "download_hlo_css",
    src = "download_hlo.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
