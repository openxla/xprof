load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = [
    "//frontend:__subpackages__",
    "//frontend:internal",
])

xprof_ng_module(
    name = "memory_viewer_control",
    srcs = [
        "memory_viewer_control.ts",
        "memory_viewer_control_module.ts",
    ],
    assets = [
        ":memory_viewer_control_css",
        "memory_viewer_control.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/controls/download_hlo",
        "//frontend/app/components/controls/searchable_dropdown",
    ],
)

sass_binary(
    name = "memory_viewer_control_css",
    src = "memory_viewer_control.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
