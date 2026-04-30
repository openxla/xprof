load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "stats_table",
    srcs = [
        "stats_table.ts",
        "stats_table_module.ts",
    ],
    assets = [
        ":stats_table_css",
        "stats_table.ng.html",
    ],
    # strict_templates = False,
    deps = [
        ":stats_table_data_provider",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart",
    ],
)

xprof_ng_module(
    name = "stats_table_data_provider",
    srcs = [
        "stats_table_data_provider.ts",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/components/chart:default_data_provider",
        "@org_xprof//frontend/app/components/chart:utils",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "stats_table_test.ts",
    ],
    deps = [
        ":stats_table",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/utils:testing",
    ],
)

sass_binary(
    name = "stats_table_css",
    src = "stats_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
