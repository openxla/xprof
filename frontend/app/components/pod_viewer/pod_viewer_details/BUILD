load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "pod_viewer_details",
    srcs = [
        "pod_viewer_details.ts",
        "pod_viewer_details_module.ts",
    ],
    assets = [
        ":pod_viewer_details_css",
        "pod_viewer_details.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "pod_viewer_details_test.ts",
    ],
    deps = [
        ":pod_viewer_details",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "pod_viewer_details_css",
    src = "pod_viewer_details.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
