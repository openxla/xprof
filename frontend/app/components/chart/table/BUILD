load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
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
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "//third_party/javascript/angular_components:cdk_testing_catalyst",
        "@org_xprof//frontend/app/common/angular:angular_material_select_testing",
        "//:node_modules/rxjs",
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
