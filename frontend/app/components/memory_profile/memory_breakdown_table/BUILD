load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:memory_profile_proto_defs",
        "//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "memory_breakdown_table_css",
    src = "memory_breakdown_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
