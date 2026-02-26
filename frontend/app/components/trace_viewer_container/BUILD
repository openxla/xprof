load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")
load("//javascript/angular2:build_defs.bzl", "ng_module")

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
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@npm//angular-split",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_spinner",
        "@org_xprof//frontend/app/common/angular:angular_material_table",
        "@org_xprof//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "@org_xprof//frontend/app/pipes",
    ],
)

ng_module(
    name = "trace_viewer_container_tests",
    testonly = True,
    srcs = ["trace_viewer_container_test.ts"],
    deps = [
        ":trace_viewer_container",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
    ],
)

sass_binary(
    name = "trace_viewer_container_css",
    src = "trace_viewer_container.scss",
    sourcemap = False,
)
