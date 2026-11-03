load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "trace_viewer",
    srcs = [
        "constants.ts",
        "filter_chips.ts",
        "filter_input.ts",
        "interfaces.ts",
        "trace_viewer.ts",
        "trace_viewer_module.ts",
        "trace_viewer_typings.ts",
        "utils.ts",
    ],
    assets = [
        ":trace_viewer_css",
        "trace_viewer.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_common_http",
        "//frontend/app/common/angular:angular_material_autocomplete",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_chips",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_menu",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/components/trace_viewer_container",
        "//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "//frontend/app/components/trace_viewer_v2/trace_helper:event_args_keys",
        "//frontend/app/pipes",
        "//frontend/app/services/communication_service",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "trace_viewer_css",
    src = "trace_viewer.scss",
    # stack = False,
    sourcemap = False,
)
