load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "capture_profile_dialog",
    srcs = [
        "capture_profile_dialog.ts",
        "capture_profile_dialog_module.ts",
    ],
    assets = [
        ":capture_profile_dialog_css",
        "capture_profile_dialog.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/platform-browser",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_radio",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "capture_profile_dialog_css",
    src = "capture_profile_dialog.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
