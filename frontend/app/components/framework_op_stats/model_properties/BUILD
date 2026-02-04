load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "model_properties",
    srcs = [
        "model_properties.ts",
        "model_properties_module.ts",
    ],
    assets = [
        ":model_properties_css",
        "model_properties.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@types/google.visualization",
    ],
)

sass_binary(
    name = "model_properties_css",
    src = "model_properties.scss",
    # stack = False,
    sourcemap = False,
)
