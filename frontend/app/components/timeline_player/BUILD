load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "timeline_player",
    srcs = [
        "time_format.pipe.ts",
        "timeline_player.ts",
    ],
    assets = [
        ":timeline_player_css",
        "timeline_player.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_menu",
    ],
)

sass_binary(
    name = "timeline_player_css",
    src = "timeline_player.scss",
    sourcemap = False,
)
