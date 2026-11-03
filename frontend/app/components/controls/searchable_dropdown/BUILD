load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = [
    "//frontend:__subpackages__",
    "//labs:__subpackages__",
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
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_select",
    ],
)

sass_binary(
    name = "searchable_dropdown_css",
    src = "searchable_dropdown.scss",
    # stack = True,
    sourcemap = False,
)
