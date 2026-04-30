load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "empty_page",
    srcs = [
        "empty_page.ts",
        "empty_page_module.ts",
    ],
    assets = [
        ":empty_page_css",
        "empty_page.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/capture_profile",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "empty_page_test.ts",
    ],
    deps = [
        ":empty_page",
        "//javascript/angular2/testing/catalyst/fake_async",
        # ksf,
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@npm//rxjs",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "empty_page_css",
    src = "empty_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
