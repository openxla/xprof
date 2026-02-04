load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/components/op_profile/op_table_entry",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "op_table_css",
    src = "op_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/components/op_profile:op_profile_common",
    ],
)
