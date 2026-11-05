load("//defs:defs.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "testing",
    srcs = [
        "testing.ts",
    ],
    deps = [
        "//frontend/app/common/constants:testing",
    ],
)

ts_library(
    name = "utils",
    srcs = [
        "diff_utils.ts",
        "serving_metadata_utils.ts",
        "utils.ts",
    ],
    deps = [
        "//:node_modules/@ngrx/store",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/store",
    ],
)

ts_library(
    name = "chart_utils",
    srcs = [
        "chart_utils.ts",
    ],
    deps = [
        "//:node_modules/@types/google.visualization",
    ],
)
