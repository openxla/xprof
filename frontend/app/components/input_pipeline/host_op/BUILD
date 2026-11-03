load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "host_op",
    srcs = [
        "host_op.ts",
        "host_op_module.ts",
    ],
    assets = [
        ":host_op_css",
        "host_op.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
        "//frontend/app/components/chart:default_data_provider",
    ],
)

sass_binary(
    name = "host_op_css",
    src = "host_op.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
