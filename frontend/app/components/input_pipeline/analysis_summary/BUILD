load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "analysis_summary",
    srcs = [
        "analysis_summary.ts",
        "analysis_summary_module.ts",
    ],
    assets = [
        ":analysis_summary_css",
        "analysis_summary.ng.html",
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
        "analysis_summary_test.ts",
    ],
    deps = [
        ":analysis_summary",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "analysis_summary_css",
    src = "analysis_summary.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
