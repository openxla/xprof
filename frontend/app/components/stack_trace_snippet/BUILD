load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/styles:common",
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
        "//frontend/app/common/angular:angular_common_http",
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_tooltip",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/pipes",
        "//frontend/app/services/communication_service",
        "//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "stack_frame_snippet_css",
    src = "stack_frame_snippet.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
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
        "//frontend/app/common/angular:angular_material_expansion",
        "//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "stack_trace_snippet_css",
    src = "stack_trace_snippet.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
