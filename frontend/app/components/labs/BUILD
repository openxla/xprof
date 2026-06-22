load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//visibility:public"])

xprof_ng_module(
    name = "labs",
    srcs = [
        "curated_tool.ts",
        "curated_tools.service.ts",
        "labs.ts",
        "labs_module.ts",
    ],
    assets = [
        ":labs_css",
        "labs.ng.html",
    ],
    deps = [
        "//third_party/javascript/safevalues/dom",
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/core_rxjs_interop",
        "@npm//@angular/forms",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@npm//safevalues",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_button_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/angular:angular_material_chips",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_snack_bar",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "labs_css",
    src = "labs.scss",
    # stack = True,
    sourcemap = False,
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "curated_tools_service_test.ts",
        "labs_test.ts",
    ],
    deps = [
        ":labs",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/safevalues/dom",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@npm//@angular/core_testing",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@npm//safevalues",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/angular:angular_common_http_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_input_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_snack_bar",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/store",
    ],
)
