load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = [
    "@org_xprof//frontend:__subpackages__",
])

xprof_ng_module(
    name = "max_heap_chart_downloader",
    srcs = [
        "max_heap_chart_downloader.ts",
        "max_heap_chart_downloader_module.ts",
    ],
    assets = [
        ":max_heap_chart_downloader_css",
        "max_heap_chart_downloader.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/memory_viewer/memory_usage",
    ],
)

sass_binary(
    name = "max_heap_chart_downloader_css",
    src = "max_heap_chart_downloader.scss",
    # stack = True,
    sourcemap = False,
    deps = [
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "max_heap_chart_downloader_test.ts",
    ],
    visibility = ["//frontend:internal"],
    deps = [
        ":max_heap_chart_downloader",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//javascript/security:safe_downloader",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)
