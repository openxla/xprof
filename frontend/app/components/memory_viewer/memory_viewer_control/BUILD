load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = [
    "//frontend:internal",
    "@org_xprof//frontend:__subpackages__",
])

xprof_ng_module(
    name = "memory_viewer_control",
    srcs = [
        "memory_viewer_control.ts",
        "memory_viewer_control_module.ts",
    ],
    assets = [
        ":memory_viewer_control_css",
        "memory_viewer_control.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/controls/download_hlo",
        "@org_xprof//frontend/app/components/controls/searchable_dropdown",
    ],
)

sass_binary(
    name = "memory_viewer_control_css",
    src = "memory_viewer_control.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
