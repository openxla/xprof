load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "program_order_chart",
    srcs = [
        "program_order_chart.ts",
        "program_order_chart_module.ts",
    ],
    assets = [
        ":program_order_chart_css",
        "program_order_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "program_order_chart_test.ts",
    ],
    deps = [
        ":program_order_chart",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "program_order_chart_css",
    src = "program_order_chart.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
