load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "table",
    srcs = [
        "table.ts",
        "table_module.ts",
    ],
    assets = [
        ":table_css",
        "table.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "table_test.ts",
    ],
    deps = [
        ":table",
        "//javascript/angular2/testing/catalyst/fake_async",
        # ksf,
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "@org_xprof//frontend/app/common/angular:angular_material_select_testing",
        "@npm//rxjs",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "table_css",
    src = "table.scss",
    # stack = False,
    sourcemap = False,
)
