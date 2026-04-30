load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "kernel_stats_table",
    srcs = [
        "kernel_stats_table.ts",
        "kernel_stats_table_module.ts",
    ],
    assets = [
        ":kernel_stats_table_css",
        "kernel_stats_table.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "kernel_stats_table_test.ts",
    ],
    deps = [
        ":kernel_stats_table",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_input_testing",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "kernel_stats_table_css",
    src = "kernel_stats_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
