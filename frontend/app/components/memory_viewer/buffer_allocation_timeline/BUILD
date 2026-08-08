load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "buffer_allocation_timeline",
    srcs = [
        "buffer_allocation_timeline.ts",
        "buffer_allocation_timeline_module.ts",
    ],
    assets = [
        ":buffer_allocation_timeline_css",
        "buffer_allocation_timeline.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "buffer_allocation_timeline_css",
    src = "buffer_allocation_timeline.scss",
    sourcemap = False,
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "buffer_allocation_timeline_test.ts",
    ],
    deps = [
        ":buffer_allocation_timeline",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)
