load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "smart_suggestion",
    srcs = [
        "smart_suggestion_view.ts",
    ],
    assets = [
        ":smart_suggestion_view_css",
        "smart_suggestion_view.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "smart_suggestion_view_css",
    src = "smart_suggestion_view.scss",
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

xprof_ng_module(
    name = "smart_suggestion_tests",
    testonly = True,
    srcs = ["smart_suggestion_test.ts"],
    deps = [
        ":smart_suggestion",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/typings/jasmine",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion_testing",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)
