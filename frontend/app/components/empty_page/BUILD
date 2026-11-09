load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/rxjs",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/capture_profile",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

sass_binary(
    name = "empty_page_css",
    src = "empty_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
