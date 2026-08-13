load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
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
        "//:node_modules/@angular/core",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
    ],
)

sass_binary(
    name = "model_properties_css",
    src = "model_properties.scss",
    # stack = False,
    sourcemap = False,
)
