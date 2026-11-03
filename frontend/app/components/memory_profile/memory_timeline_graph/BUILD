load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_timeline_graph",
    srcs = [
        "memory_timeline_graph.ts",
        "memory_timeline_graph_module.ts",
    ],
    assets = [
        ":memory_timeline_graph_css",
        "memory_timeline_graph.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
    ],
)

sass_binary(
    name = "memory_timeline_graph_css",
    src = "memory_timeline_graph.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
