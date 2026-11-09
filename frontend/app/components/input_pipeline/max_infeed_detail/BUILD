load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "max_infeed_detail",
    srcs = [
        "max_infeed_detail.ts",
        "max_infeed_detail_module.ts",
    ],
    assets = [
        ":max_infeed_detail_css",
        "max_infeed_detail.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "max_infeed_detail_css",
    src = "max_infeed_detail.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
