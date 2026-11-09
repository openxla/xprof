load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "trace_viewer_container",
    srcs = [
        "hlo_pretty_printer.ts",
        "trace_viewer_container.ts",
        "trace_viewer_utils.ts",
    ],
    assets = [
        ":trace_viewer_container_css",
        "trace_viewer_container.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/router",
        "//:node_modules/angular-split",
        "//:node_modules/ngx-json-viewer",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_menu",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_progress_spinner",
        "//frontend/app/common/angular:angular_material_sort",
        "//frontend/app/common/angular:angular_material_table",
        "//frontend/app/common/angular:angular_material_tabs",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/static_kernel_viewer:kernel_minimap",
        "//frontend/app/components/timeline_player",
        "//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "//frontend/app/pipes",
    ],
)

sass_binary(
    name = "trace_viewer_container_css",
    src = "trace_viewer_container.scss",
    sourcemap = False,
)
