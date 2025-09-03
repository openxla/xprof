load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "sass_library")
load("@npm//@bazel/concatjs:index.bzl", "ts_library")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "op_profile",
    srcs = [
        "op_profile.ts",
        "op_profile_module.ts",
    ],
    assets = [
        ":op_profile_common_css",
        "op_profile.ng.html",
    ],
    # strict_templates = False,
    deps = [
        ":op_profile_base",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_sidenav",
        "@org_xprof//frontend/app/common/angular:angular_material_slide_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/op_profile/op_details",
        "@org_xprof//frontend/app/components/op_profile/op_table",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "op_profile_base",
    srcs = [
        "op_profile_base.ts",
        "op_profile_base_module.ts",
    ],
    assets = [
        "op_profile_base.ng.html",
        ":op_profile_common_css",
    ],
    deps = [
        ":op_profile_data",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//angular-split",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/angular:angular_material_sidenav",
        "@org_xprof//frontend/app/common/angular:angular_material_slide_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/op_profile/op_table",
        "@org_xprof//frontend/app/components/stack_trace_snippet",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "op_profile_common_css",
    src = "op_profile_common.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

sass_library(
    name = "op_profile_common",
    srcs = ["op_profile_common.scss"],
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

ts_library(
    name = "op_profile_data",
    srcs = [
        "op_profile_data.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/common/utils",
    ],
)
