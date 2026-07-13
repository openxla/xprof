load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "profiler_options",
    srcs = ["profiler_options.component.ts"],
    assets = [
        "profiler_options.component.css",
        "profiler_options.component.html",
    ],
    visibility = ["@org_xprof//frontend/app/components/kernel_analysis:__pkg__"],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_button_toggle",
        "@org_xprof//frontend/app/common/angular:angular_material_checkbox",
        "@org_xprof//frontend/app/common/angular:angular_material_dialog",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_radio",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/components/kernel_analysis/data",
        "@org_xprof//frontend/app/components/kernel_analysis/shared/components/counter_selection",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "profiler_options.component_test.ts",
    ],
    deps = [
        ":profiler_options",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/forms",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_dialog",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/components/kernel_analysis/data",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)
