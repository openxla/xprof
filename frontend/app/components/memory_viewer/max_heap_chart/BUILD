load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "max_heap_chart",
    srcs = [
        "max_heap_chart.ts",
        "max_heap_chart_module.ts",
    ],
    allow_warnings = True,
    assets = [
        ":max_heap_chart_css",
        "max_heap_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "max_heap_chart_test.ts",
    ],
    deps = [
        ":max_heap_chart",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "max_heap_chart_css",
    src = "max_heap_chart.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
