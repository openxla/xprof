load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "step_time_graph",
    srcs = [
        "step_time_graph.ts",
        "step_time_graph_module.ts",
    ],
    assets = [
        ":step_time_graph_css",
        "step_time_graph.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils:chart_utils",
    ],
)

sass_binary(
    name = "step_time_graph_css",
    src = "step_time_graph.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
