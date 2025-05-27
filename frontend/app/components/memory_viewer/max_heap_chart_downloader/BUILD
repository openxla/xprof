load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//javascript/angular2:build_defs.bzl", "ng_module")

package(default_visibility = ["//perftools/accelerators/xprof/frontend:__subpackages__"])

ng_module(
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
        "//third_party/javascript/ts_csv",
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/controls/download_hlo",
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
