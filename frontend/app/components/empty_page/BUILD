load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

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
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/capture_profile",
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
