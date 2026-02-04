load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "table",
    srcs = [
        "table.ts",
        "table_module.ts",
    ],
    assets = [
        ":table_css",
        "table.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
    ],
)

sass_binary(
    name = "table_css",
    src = "table.scss",
    # stack = False,
    sourcemap = False,
)
