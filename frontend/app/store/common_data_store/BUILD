load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "common_data_store",
    srcs = [
        "actions.ts",
        "reducers.ts",
        "selectors.ts",
        "state.ts",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
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
        ":common_data_store",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
    ],
)
