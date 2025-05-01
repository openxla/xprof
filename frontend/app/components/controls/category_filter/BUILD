load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "category_filter",
    srcs = [
        "category_filter.ts",
        "category_filter_module.ts",
    ],
    assets = [
        "category_filter.ng.html",
        ":category_filter_css",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
    ],
)

sass_binary(
    name = "category_filter_css",
    src = "category_filter.scss",
    # stack = False,
    sourcemap = False,
)
