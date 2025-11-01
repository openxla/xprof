load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "source_mapper",
    srcs = [
        "source_mapper.ts",
        "source_mapper_module.ts",
    ],
    assets = [
        ":source_mapper_css",
        "source_mapper.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/components/stack_trace_snippet",
        "@org_xprof//frontend/app/components/stack_trace_snippet:message",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "source_mapper_css",
    src = "source_mapper.scss",
    # stack = True,
    sourcemap = False,
    deps = ["@org_xprof//frontend/app/styles:common"],
)
