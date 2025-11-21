load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "stack_trace_page",
    srcs = [
        "stack_trace_page.ts",
        "stack_trace_page_module.ts",
    ],
    assets = [
        ":stack_trace_page_css",
        "stack_trace_page.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/components/source_mapper",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "stack_trace_page_css",
    src = "stack_trace_page.scss",
    # stack = True,
    sourcemap = False,
    deps = ["@org_xprof//frontend/app/styles:common"],
)
