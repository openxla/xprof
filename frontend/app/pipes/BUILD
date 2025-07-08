load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "pipes",
    srcs = [
        "pipes_module.ts",
        "safe_pipe.ts",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/platform-browser",
    ],
)
