load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "data_service_v2",
    srcs = [
        "data_service_v2.ts",
    ],
    deps = [
        ":data_service_v2_interface",
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//:node_modules/safevalues",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/op_profile:op_profile_data",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "data_service_v2_interface",
    srcs = [
        "data_service_v2_interface.ts",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/op_profile:op_profile_data",
    ],
)
