load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "view_architecture",
    srcs = [
        "view_architecture.ts",
        "view_architecture_module.ts",
    ],
    assets = [
        "view_architecture.ng.html",
        ":view_architecture_css",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = ["view_architecture_test.ts"],
    deps = [
        ":view_architecture",
        "//:node_modules/rxjs",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "view_architecture_css",
    src = "view_architecture.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
