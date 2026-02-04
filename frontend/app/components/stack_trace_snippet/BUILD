load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "message",
    srcs = [
        "message.ts",
    ],
    assets = [
        ":message_css",
        "message.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
    ],
)

sass_binary(
    name = "message_css",
    src = "message.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

xprof_ng_module(
    name = "stack_frame_snippet",
    srcs = [
        "stack_frame_snippet.ts",
        "stack_frame_snippet_module.ts",
    ],
    assets = [
        ":stack_frame_snippet_css",
        "stack_frame_snippet.ng.html",
    ],
    deps = [
        ":message",
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/pipes",
        "@org_xprof//frontend/app/services/communication_service",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "stack_frame_snippet_css",
    src = "stack_frame_snippet.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

xprof_ng_module(
    name = "stack_trace_snippet",
    srcs = [
        "stack_trace_snippet.ts",
        "stack_trace_snippet_module.ts",
    ],
    assets = [
        ":stack_trace_snippet_css",
        "stack_trace_snippet.ng.html",
    ],
    deps = [
        ":message",
        ":stack_frame_snippet",
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "stack_trace_snippet_css",
    src = "stack_trace_snippet.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
