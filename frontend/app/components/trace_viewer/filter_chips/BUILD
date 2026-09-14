load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "filter_chips",
    srcs = ["filter_chips.ts"],
    assets = [
        ":filter_chips_css",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_autocomplete",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_chips",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/components/trace_viewer:trace_viewer_typings",
    ],
)

sass_binary(
    name = "filter_chips_css",
    src = "filter_chips.scss",
    # stack = True,
    sourcemap = False,
)
