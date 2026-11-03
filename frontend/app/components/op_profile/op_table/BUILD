load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "op_table",
    srcs = [
        "op_table.ts",
        "op_table_module.ts",
    ],
    assets = [
        ":op_table_css",
        "op_table.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/components/op_profile/op_table_entry",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "op_table_css",
    src = "op_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/components/op_profile:op_profile_common",
    ],
)
