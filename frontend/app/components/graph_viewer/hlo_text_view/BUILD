load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "hlo_text_view",
    srcs = [
        "hlo_text_view.ts",
        "hlo_text_view_module.ts",
    ],
    assets = [
        ":hlo_text_view_css",
        "hlo_text_view.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = ["hlo_text_view_test.ts"],
    deps = [
        ":hlo_text_view",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/router",
        "@npm//@angular/router_testing",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion_testing",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/utils:testing",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "hlo_text_view_css",
    src = "hlo_text_view.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
