load("//defs:defs.bzl", "ts_library", "xprof_ng_module")

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
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:op_profile_proto_defs",
        "//frontend/app/store/common_data_store",
        "//frontend/app/store/framework_op_stats",
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
