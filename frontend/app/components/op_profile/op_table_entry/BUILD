load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "op_table_entry",
    srcs = [
        "op_table_entry.ts",
        "op_table_entry_module.ts",
    ],
    assets = [
        ":op_table_entry_css",
        "op_table_entry.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "op_table_entry_css",
    src = "op_table_entry.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/components/op_profile:op_profile_common",
    ],
)
