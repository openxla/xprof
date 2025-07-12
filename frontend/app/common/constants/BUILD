load("//defs:defs.bzl", ts_library = "xprof_ng_module")

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
        "//:node_modules/@types/google.visualization",
    ],
)

ts_library(
    name = "testing",
    srcs = [
        "testing.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces",
    ],
)
