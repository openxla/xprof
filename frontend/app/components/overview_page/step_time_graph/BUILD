load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "step_time_graph",
    srcs = [
        "step_time_graph.ts",
        "step_time_graph_module.ts",
    ],
    assets = [
        ":step_time_graph_css",
        "step_time_graph.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "step_time_graph_test.ts",
    ],
    deps = [
        ":step_time_graph",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "step_time_graph_css",
    src = "step_time_graph.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
