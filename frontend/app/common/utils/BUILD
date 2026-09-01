load("//defs:defs.bzl", "ts_library")

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
        "diff_utils.ts",
        "utils.ts",
    ],
    deps = [
        "//:node_modules/@ngrx/store",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/store",
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

ts_library(
    name = "tests",
    testonly = True,
    srcs = [
        "chart_utils_test.ts",
        "diff_utils_test.ts",
        "utils_test.ts",
    ],
    deps = [
        ":chart_utils",
        ":testing",
        ":utils",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
    ],
)
