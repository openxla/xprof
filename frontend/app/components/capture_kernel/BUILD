load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "capture_kernel",
    srcs = [
        "capture_kernel.ts",
    ],
    assets = [
        ":capture_kernel_css",
        "capture_kernel.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/components/capture_kernel/capture_kernel_dialog",
    ],
)

sass_binary(
    name = "capture_kernel_css",
    src = "capture_kernel.scss",
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
