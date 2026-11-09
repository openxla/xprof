load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "inference_latency_chart",
    srcs = [
        "inference_latency_chart.ts",
        "inference_latency_chart_module.ts",
    ],
    assets = [
        ":inference_latency_chart_css",
        "inference_latency_chart.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
        "//frontend/app/common/angular:angular_material_card",
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "inference_latency_chart_css",
    src = "inference_latency_chart.scss",
    # stack = True,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
