load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "tb_combine_html", "tf_web_library")

package(default_visibility = ["//plugin:internal"])

licenses(["notice"])

tf_web_library(
    name = "trace_viewer",
    srcs = ["trace_viewer.html"],
    path = "/",
    deps = [
        "@npm//@polymer/polymer",
        "@org_xprof//plugin/trace_viewer/tf_trace_viewer",
        "@org_xprof//plugin/trace_viewer/webcomponentsjs_polyfill",
    ],
)

tb_combine_html(
    name = "trace_viewer_index",
    input_path = "/trace_viewer.html",
    js_path = "/trace_viewer_index.js",
    output_path = "/trace_viewer_index.html",
    deps = [":trace_viewer"],
)
