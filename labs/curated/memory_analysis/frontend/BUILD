load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//visibility:public"])

xprof_ng_module(
    name = "memory_analysis",
    srcs = [
        "flame_graph/flame_graph.ts",
        "memory_analysis.ts",
        "memory_analysis_module.ts",
        "treemap/treemap.ts",
        "utils/tree_builder.ts",
    ],
    assets = [
        ":flame_graph_css",
        ":memory_analysis_css",
        ":treemap_css",
        "flame_graph/flame_graph.ng.html",
        "memory_analysis.ng.html",
        "treemap/treemap.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/core_rxjs_interop",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_button_toggle",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_progress_spinner",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/angular:angular_material_sort",
        "//frontend/app/common/angular:angular_material_table",
        "//frontend/app/common/angular:angular_material_tabs",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/controls/searchable_dropdown",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "memory_analysis_css",
    src = "memory_analysis.scss",
    # stack = True,
    sourcemap = False,
)

sass_binary(
    name = "flame_graph_css",
    src = "flame_graph/flame_graph.scss",
    # stack = True,
    sourcemap = False,
)

sass_binary(
    name = "treemap_css",
    src = "treemap/treemap.scss",
    # stack = True,
    sourcemap = False,
)
