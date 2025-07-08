load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_breakdown_table",
    srcs = [
        "memory_breakdown_table.ts",
        "memory_breakdown_table_module.ts",
    ],
    assets = [
        ":memory_breakdown_table_css",
        "memory_breakdown_table.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:memory_profile_proto_defs",
        "@org_xprof//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "memory_breakdown_table_css",
    src = "memory_breakdown_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
