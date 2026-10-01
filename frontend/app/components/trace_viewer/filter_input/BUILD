load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "filter_input",
    srcs = ["filter_input.ts"],
    assets = [
        ":filter_input_css",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_autocomplete",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/components/trace_viewer:constants",
        "@org_xprof//frontend/app/components/trace_viewer:trace_viewer_typings",
        "@org_xprof//frontend/app/components/trace_viewer:utils",
    ],
)

sass_binary(
    name = "filter_input_css",
    src = "filter_input.scss",
    # stack = True,
    sourcemap = False,
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = ["filter_input_test.ts"],
    deps = [
        ":filter_input",
        "//:node_modules/@angular/core",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_material_core_testing",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/components/trace_viewer:constants",
        "@org_xprof//frontend/app/components/trace_viewer:trace_viewer_typings",
    ],
)
