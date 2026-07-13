load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_dialog",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_list",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
    ],
)

sass_binary(
    name = "counter_selection_css",
    src = "counter_selection.component.scss",
    # stack = True,
    sourcemap = False,
)
