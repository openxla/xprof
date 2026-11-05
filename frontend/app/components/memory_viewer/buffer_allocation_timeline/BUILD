load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "buffer_allocation_timeline_css",
    src = "buffer_allocation_timeline.scss",
    sourcemap = False,
)
