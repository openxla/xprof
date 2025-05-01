load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "syntax_highlighter",
    srcs = [
        "syntax_highlight_service.ts",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//third_party/javascript/highlightjs:highlightjs_bash",
        "//third_party/javascript/highlightjs:highlightjs_c",
        "//third_party/javascript/highlightjs:highlightjs_cpp",
        "//third_party/javascript/highlightjs:highlightjs_css",
        "//third_party/javascript/highlightjs:highlightjs_go",
        "//third_party/javascript/highlightjs:highlightjs_html",
        "//third_party/javascript/highlightjs:highlightjs_java",
        "//third_party/javascript/highlightjs:highlightjs_kotlin",
        "//third_party/javascript/highlightjs:highlightjs_python",
        "//third_party/javascript/highlightjs:highlightjs_raw",
        "//third_party/javascript/highlightjs:highlightjs_sql",
        "//third_party/javascript/highlightjs:highlightjs_typescript",
    ],
)

xprof_ng_module(
    name = "source_code_service_interface",
    srcs = [
        "source_code_service_interface.ts",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/rxjs",
    ],
)

xprof_ng_module(
    name = "source_code_service",
    srcs = [
        "source_code_service.ts",
    ],
    deps = [
        ":source_code_service_interface",
        "//:node_modules/rxjs",
    ],
)
