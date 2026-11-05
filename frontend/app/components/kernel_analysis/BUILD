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
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_stepper",
        "//frontend/app/components/kernel_analysis/data",
        "//frontend/app/components/kernel_analysis/execution_preview",
        "//frontend/app/components/kernel_analysis/profiler_options",
    ],
)
