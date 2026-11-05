load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//visibility:public"])

xprof_ng_module(
    name = "labs",
    srcs = [
        "curated_tool.ts",
        "curated_tools.service.ts",
        "labs.ts",
        "labs_module.ts",
    ],
    assets = [
        ":labs_css",
        "labs.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/core_rxjs_interop",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//:node_modules/safevalues",
        "//frontend/app/common/angular:angular_common_http",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_button_toggle",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_chips",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_snack_bar",
        "//frontend/app/store",
        "//third_party/javascript/safevalues/dom",
    ],
)

sass_binary(
    name = "labs_css",
    src = "labs.scss",
    # stack = True,
    sourcemap = False,
)
