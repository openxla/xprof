load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "trace_viewer_container",
    srcs = [
        "trace_viewer_container.ts",
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
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_menu",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_spinner",
        "@org_xprof//frontend/app/common/angular:angular_material_sort",
        "@org_xprof//frontend/app/common/angular:angular_material_table",
        "@org_xprof//frontend/app/common/angular:angular_material_tabs",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "@org_xprof//frontend/app/pipes",
    ],
)

xprof_ng_module(
    name = "trace_viewer_container_tests",
    testonly = True,
    srcs = ["trace_viewer_container_test.ts"],
    deps = [
        ":trace_viewer_container",
        "//:node_modules/@angular/platform-browser",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/components/trace_viewer_v2:app_bundle_ts",
    ],
)

sass_binary(
    name = "trace_viewer_container_css",
    src = "trace_viewer_container.scss",
    sourcemap = False,
)
