load("//defs:defs.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "dashboard",
    srcs = [
        "dashboard.ts",
    ],
    deps = [
        "//:node_modules/@types/google.visualization",
        "//frontend/app/common/interfaces",
    ],
)
