load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/components/kernel_analysis",
    ],
)

sass_binary(
    name = "capture_kernel_dialog_css",
    src = "capture_kernel_dialog.scss",
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
