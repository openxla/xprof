load("@npm//@bazel/concatjs:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "testing",
    srcs = [
        "testing.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/constants:testing",
    ],
)

ts_library(
    name = "utils",
    srcs = [
        "utils.ts",
    ],
    deps = [
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/store",
    ],
)
