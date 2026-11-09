load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "topology_graph",
    srcs = [
        "topology_graph.ts",
        "topology_graph_module.ts",
    ],
    assets = [
        ":topology_graph_css",
        "topology_graph.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_menu",
        "//frontend/app/common/angular:angular_material_slider",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "topology_graph_css",
    src = "topology_graph.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
