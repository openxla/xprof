load("//defs:defs.bzl", ts_library = "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "dashboard",
    srcs = [
        "dashboard.ts",
    ],
    deps = [
        "//:node_modules/@types/google.visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)
