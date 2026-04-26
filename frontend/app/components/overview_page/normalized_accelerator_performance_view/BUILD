load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "normalized_accelerator_performance_view",
    srcs = [
        "normalized_accelerator_performance_view.ts",
        "normalized_accelerator_performance_view_module.ts",
    ],
    assets = [
        ":normalized_accelerator_performance_view_css",
        "normalized_accelerator_performance_view.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "normalized_accelerator_performance_view_test.ts",
    ],
    deps = [
        ":normalized_accelerator_performance_view",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "normalized_accelerator_performance_view_css",
    src = "normalized_accelerator_performance_view.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
