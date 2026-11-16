load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

# DCN Analyzer page (tool `grpc_metrics`).
xprof_ng_module(
    name = "grpc_metrics",
    srcs = [
        "grpc_metrics.ts",
        "grpc_metrics_charts.ts",
        "grpc_metrics_module.ts",
        "grpc_metrics_options.ts",
        "grpc_metrics_utils.ts",
        "types.ts",
    ],
    assets = [
        ":grpc_metrics_css",
        "grpc_metrics.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/core_rxjs_interop",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_chips",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_paginator",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_spinner",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/angular:angular_material_sort",
        "@org_xprof//frontend/app/common/angular:angular_material_table",
        "@org_xprof//frontend/app/common/angular:angular_material_tabs",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "grpc_metrics_css",
    src = "grpc_metrics.scss",
    # stack = True,
    sourcemap = False,
)
