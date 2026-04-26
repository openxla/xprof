load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "framework_op_stats",
    srcs = [
        "actions.ts",
        "reducers.ts",
        "selectors.ts",
        "state.ts",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store:types",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "reducers_test.ts",
        "selectors_test.ts",
    ],
    deps = [
        ":framework_op_stats",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@ngrx/store",
        "@npm//rxjs",
    ],
)
