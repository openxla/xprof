load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "model_properties",
    srcs = [
        "model_properties.ts",
        "model_properties_module.ts",
    ],
    assets = [
        ":model_properties_css",
        "model_properties.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "model_properties_test.ts",
    ],
    deps = [
        ":model_properties",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
    ],
)

sass_binary(
    name = "model_properties_css",
    src = "model_properties.scss",
    # stack = False,
    sourcemap = False,
)
