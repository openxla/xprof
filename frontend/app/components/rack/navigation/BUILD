load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

licenses(["notice"])

xprof_ng_module(
    name = "navigation",
    srcs = [
        "index.ts",
        "navigation_drawer.ts",
        "navigation_group.ts",
        "navigation_item.ts",
    ],
    assets = [
        ":navigation_drawer_css",
        ":navigation_group_css",
        ":navigation_item_css",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
    ],
)

sass_binary(
    name = "navigation_drawer_css",
    src = "navigation_drawer.scss",
    # stack = True,
    sourcemap = False,
)

sass_binary(
    name = "navigation_group_css",
    src = "navigation_group.scss",
    # stack = True,
    sourcemap = False,
)

sass_binary(
    name = "navigation_item_css",
    src = "navigation_item.scss",
    # stack = True,
    sourcemap = False,
)
