load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "profiler_options",
    srcs = ["profiler_options.component.ts"],
    assets = [
        "profiler_options.component.css",
        "profiler_options.component.html",
    ],
    visibility = ["//frontend/app/components/kernel_analysis:__pkg__"],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_button_toggle",
        "//frontend/app/common/angular:angular_material_checkbox",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_radio",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/components/kernel_analysis/data",
        "//frontend/app/components/kernel_analysis/shared/components/counter_selection",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
    ],
)
