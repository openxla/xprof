load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/components/source_mapper",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "stack_trace_page_css",
    src = "stack_trace_page.scss",
    # stack = True,
    sourcemap = False,
    deps = ["//frontend/app/styles:common"],
)
