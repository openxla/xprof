load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/core_rxjs_interop",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_button_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_spinner",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/angular:angular_material_sort",
        "@org_xprof//frontend/app/common/angular:angular_material_table",
        "@org_xprof//frontend/app/common/angular:angular_material_tabs",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/controls/searchable_dropdown",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
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

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "memory_analysis_test.ts",
    ],
    deps = [
        ":memory_analysis",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/angular_components:cdk_testing",
        "//third_party/javascript/angular_components:cdk_testing_testbed",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/common",
        "@npm//@angular/core_testing",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_card_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_input_testing",
        "@org_xprof//frontend/app/common/angular:angular_material_select_testing",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)
