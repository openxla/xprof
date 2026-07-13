load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "kernel_analysis",
    srcs = ["kernel_analysis.component.ts"],
    assets = [
        "kernel_analysis.component.css",
        "kernel_analysis.component.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_stepper",
        "@org_xprof//frontend/app/components/kernel_analysis/data",
        "@org_xprof//frontend/app/components/kernel_analysis/execution_preview",
        "@org_xprof//frontend/app/components/kernel_analysis/profiler_options",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "kernel_analysis.component_test.ts",
    ],
    deps = [
        ":kernel_analysis",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/components/kernel_analysis/data",
        "@org_xprof//frontend/app/components/kernel_analysis/execution_preview",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)
