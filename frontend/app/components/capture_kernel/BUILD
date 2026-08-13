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
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_dialog",
        "@org_xprof//frontend/app/components/capture_kernel/capture_kernel_dialog",
    ],
)

sass_binary(
    name = "capture_kernel_css",
    src = "capture_kernel.scss",
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
