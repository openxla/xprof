load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "counter_selection",
    srcs = [
        "counter_selection.component.ts",
        "types.ts",
    ],
    assets = [
        ":counter_selection_css",
        "counter_selection.component.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_list",
        "//frontend/app/common/angular:angular_material_tooltip",
    ],
)

sass_binary(
    name = "counter_selection_css",
    src = "counter_selection.component.scss",
    # stack = True,
    sourcemap = False,
)
