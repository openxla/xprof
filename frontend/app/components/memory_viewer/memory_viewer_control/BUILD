load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
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
        "//:node_modules/@angular/core_testing",
        "//:node_modules/@angular/platform-browser",
        "//:node_modules/@angular/router",
        "//:node_modules/rxjs",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/angular_components:cdk_testing_catalyst_async",
        "//third_party/javascript/angular_components:cdk_testing_testbed",
        "//third_party/javascript/typings/jasmine",
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
