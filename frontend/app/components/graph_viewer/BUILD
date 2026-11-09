load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "graph_viewer",
    srcs = [
        "graph_viewer.ts",
        "graph_viewer_module.ts",
    ],
    assets = [
        ":graph_viewer_css",
        "graph_viewer.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//:node_modules/safevalues",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_chips",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_progress_spinner",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/angular:angular_material_snack_bar",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/classes",
        "//frontend/app/common/constants",
        "//frontend/app/common/constants:colors",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/common/utils",
        "//frontend/app/components/controls/download_hlo",
        "//frontend/app/components/controls/searchable_dropdown",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/graph_viewer/hlo_text_view",
        "//frontend/app/components/op_profile:op_profile_data",
        "//frontend/app/components/op_profile/op_details",
        "//frontend/app/components/source_mapper",
        "//frontend/app/pipes",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "graph_viewer_css",
    src = "graph_viewer.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)

filegroup(
    name = "scuba_goldens",
    srcs = ["//frontend/app/components/graph_viewer/scuba_goldens"],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/*.html"]),
)
