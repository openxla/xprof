load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "flop_rate_chart",
    srcs = [
        "flop_rate_chart.ts",
        "flop_rate_chart_module.ts",
    ],
    assets = [
        ":flop_rate_chart_css",
        "flop_rate_chart.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
        "@org_xprof//frontend/app/components/chart:chart_options",
        "@org_xprof//frontend/app/components/chart:data_processor",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "flop_rate_chart_test.ts",
    ],
    deps = [
        ":flop_rate_chart",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/components/chart:chart_options",
    ],
)

sass_binary(
    name = "flop_rate_chart_css",
    src = "flop_rate_chart.scss",
    # stack = False,
    sourcemap = False,
)
