load("//defs:defs.bzl", "ts_library", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "data_dispatcher_base",
    srcs = [
        "data_dispatcher_base.ts",
    ],
    deps = [
        ":data_request_queue",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/constants",
        "//frontend/app/store",
        "//frontend/app/store:types",
        "//frontend/app/store/common_data_store",
        "//frontend/app/store/framework_op_stats",
    ],
)

ts_library(
    name = "data_request_queue",
    srcs = [
        "data_request_queue.ts",
    ],
    deps = [
        "//frontend/app/common/constants",
        "//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "data_dispatcher",
    srcs = [
        "data_dispatcher.ts",
    ],
    deps = [
        ":data_dispatcher_base",
        ":data_request_queue",
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/constants",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)
