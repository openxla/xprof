load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "overview_page",
    srcs = [
        "overview_page_module.ts",
    ],
    assets = [
        ":overview_page_css",
        "overview_page.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/overview_page/inference_latency_chart",
        "@org_xprof//frontend/app/components/overview_page/normalized_accelerator_performance_view",
        "@org_xprof//frontend/app/components/overview_page/performance_summary",
        "@org_xprof//frontend/app/components/overview_page/run_environment_view",
        "@org_xprof//frontend/app/components/overview_page/step_time_graph",
        "@org_xprof//frontend/app/components/smart_suggestion",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "overview_page_test.ts",
    ],
    deps = [
        ":overview_page",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/core_testing",
        "//:node_modules/@angular/router",
        "//:node_modules/rxjs",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/constants:testing",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils:testing",
        "@org_xprof//frontend/app/components/smart_suggestion",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "overview_page_css",
    src = "overview_page.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

xprof_ng_module(
    name = "screenshot_tests",
    testonly = True,
    srcs = [
        "overview_page_screenshot_test.ts",
    ],
    tags = ["requires-net:external"],
    deps = [
        ":overview_page",
        "//javascript/angular2/testing/catalyst/async",
        # ksf,
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "//:node_modules/@angular/router",
        "//:node_modules/rxjs",
        "//:node_modules/safevalues",
        "//third_party/javascript/safevalues/dom",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/constants:testing",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils:testing",
        "@org_xprof//frontend/app/components/smart_suggestion",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)
