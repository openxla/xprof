load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "diagnostics_view",
    srcs = [
        "diagnostics_view.ts",
        "diagnostics_view_module.ts",
    ],
    assets = [
        ":diagnostics_view_css",
        "diagnostics_view.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "diagnostics_view_test.ts",
    ],
    deps = [
        ":diagnostics_view",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "diagnostics_view_css",
    src = "diagnostics_view.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
