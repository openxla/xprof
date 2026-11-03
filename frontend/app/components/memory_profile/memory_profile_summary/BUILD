load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_profile_summary",
    srcs = [
        "memory_profile_summary.ts",
        "memory_profile_summary_module.ts",
    ],
    assets = [
        ":memory_profile_summary_css",
        "memory_profile_summary.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "memory_profile_summary_css",
    src = "memory_profile_summary.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
