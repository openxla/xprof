load("//defs:defs.bzl", "xprof_ng_module", ts_library = "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "data_service",
    srcs = [
        "data_service.ts",
    ],
    deps = [
        ":mock_data",
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store",
    ],
)

ts_library(
    name = "mock_data",
    srcs = [
        "mock_data.ts",
    ],
)
