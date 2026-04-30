load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "max_infeed_detail",
    srcs = [
        "max_infeed_detail.ts",
        "max_infeed_detail_module.ts",
    ],
    assets = [
        ":max_infeed_detail_css",
        "max_infeed_detail.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "max_infeed_detail_test.ts",
    ],
    deps = [
        ":max_infeed_detail",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "max_infeed_detail_css",
    src = "max_infeed_detail.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
