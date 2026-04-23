load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = [
    "//frontend:internal",
    "@org_xprof//frontend:__subpackages__",
])

xprof_ng_module(
    name = "memory_viewer_control",
    srcs = [
        "memory_viewer_control.ts",
        "memory_viewer_control_module.ts",
    ],
    assets = [
        ":memory_viewer_control_css",
        "memory_viewer_control.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/controls/download_hlo",
        "@org_xprof//frontend/app/components/controls/searchable_dropdown",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = ["memory_viewer_control_test.ts"],
    deps = [
        ":memory_viewer_control",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/angular_components:cdk_testing_catalyst_async",
        "//third_party/javascript/angular_components:cdk_testing_testbed",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core_testing",
        "@npm//@angular/platform-browser",
        "@npm//@angular/router",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_input_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_select_testing",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "memory_viewer_control_css",
    src = "memory_viewer_control.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
