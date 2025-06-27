load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = [
    "@org_xprof//frontend:__subpackages__",
])

xprof_ng_module(
    name = "searchable_dropdown",
    srcs = [
        "searchable_dropdown.ts",
    ],
    assets = [
        ":searchable_dropdown_css",
        "searchable_dropdown.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
    ],
)

sass_binary(
    name = "searchable_dropdown_css",
    src = "searchable_dropdown.scss",
    # stack = True,
    sourcemap = False,
)
