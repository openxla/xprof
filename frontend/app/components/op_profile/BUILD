load("//defs:defs.bzl", "sass_binary", "sass_library", "ts_library", "xprof_ng_module")

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
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/angular:angular_material_slide_toggle",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/common/utils",
        "//frontend/app/components/op_profile/op_details",
        "//frontend/app/components/op_profile/op_table",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
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
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/angular:angular_material_slide_toggle",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/common/utils",
        "//frontend/app/components/op_profile/op_table",
        "//frontend/app/components/source_mapper",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "op_profile_common_css",
    src = "op_profile_common.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)

sass_library(
    name = "op_profile_common",
    srcs = ["op_profile_common.scss"],
    deps = [
        "//frontend/app/styles:common",
    ],
)

ts_library(
    name = "op_profile_data",
    srcs = [
        "op_profile_data.ts",
    ],
    deps = [
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/common/utils",
    ],
)
