load("//defs:defs.bzl", "xprof_ng_module", ts_library = "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary", "sass_library")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/angular-split",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
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
