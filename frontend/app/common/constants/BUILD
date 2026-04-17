load("@npm//@bazel/concatjs:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "colors",
    srcs = ["colors.ts"],
)

ts_library(
    name = "constants",
    srcs = [
        "constants.ts",
        "enums.ts",
    ],
)

ts_library(
    name = "roofline_model_constants",
    srcs = [
        "roofline_model_constants.ts",
    ],
    deps = [
        "@npm//@types/google.visualization",
    ],
)

ts_library(
    name = "testing",
    srcs = [
        "testing.ts",
    ],
    deps = [
        "@npm//@types/google.visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)
