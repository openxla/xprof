load("//defs:defs.bzl", "xprof_ng_module", ts_library = "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "store",
    srcs = [
        "actions.ts",
        "reducers.ts",
        "selectors.ts",
        "state.ts",
        "store_module.ts",
    ],
    deps = [
        ":types",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/store/common_data_store",
        "@org_xprof//frontend/app/store/framework_op_stats",
    ],
)

ts_library(
    name = "types",
    srcs = [
        "types.ts",
    ],
    deps = [
        "//:node_modules/@ngrx/store",
    ],
)
