load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "capture_kernel_dialog",
    srcs = [
        "capture_kernel_dialog.ts",
    ],
    assets = [
        ":capture_kernel_dialog_css",
        "capture_kernel_dialog.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_dialog",
        "@org_xprof//frontend/app/components/kernel_analysis",
    ],
)

sass_binary(
    name = "capture_kernel_dialog_css",
    src = "capture_kernel_dialog.scss",
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
