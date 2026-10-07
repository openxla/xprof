load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

sass_binary(
    name = "static_kernel_viewer_css",
    src = "static_kernel_viewer.scss",
)

sass_binary(
    name = "kernel_event_tooltip_css",
    src = "kernel_event_tooltip.scss",
)

xprof_ng_module(
    name = "static_kernel_viewer",
    srcs = [
        "kernel_event_tooltip.ts",
        "static_kernel_viewer.ts",
    ],
    assets = [
        ":kernel_event_tooltip_css",
        ":static_kernel_viewer_css",
        "kernel_event_tooltip.ng.html",
        "static_kernel_viewer.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_spinner",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/components/trace_viewer_container",
        "@org_xprof//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "kernel_event_tooltip_test.ts",
        "static_kernel_viewer_test.ts",
    ],
    deps = [
        ":static_kernel_viewer",
        "//javascript/angular2/testing/catalyst/fake_async",
        # ksf,
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "//:node_modules/@angular/router",
        "//:node_modules/rxjs",
        "//third_party/javascript/typings/jasmine",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/trace_viewer_container",
        "@org_xprof//frontend/app/components/trace_viewer_v2:app_bundle_ts",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)
