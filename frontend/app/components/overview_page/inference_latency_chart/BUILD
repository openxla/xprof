load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "inference_latency_chart",
    srcs = [
        "inference_latency_chart.ts",
        "inference_latency_chart_module.ts",
    ],
    assets = [
        ":inference_latency_chart_css",
        "inference_latency_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "inference_latency_chart_test.ts",
    ],
    deps = [
        ":inference_latency_chart",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
    ],
)

sass_binary(
    name = "inference_latency_chart_css",
    src = "inference_latency_chart.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
