load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "run_environment_view",
    srcs = [
        "run_environment_view.ts",
        "run_environment_view_module.ts",
    ],
    assets = [
        ":run_environment_view_css",
        "run_environment_view.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "run_environment_view_css",
    src = "run_environment_view.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
