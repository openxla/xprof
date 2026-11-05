load("//defs:defs.bzl", "xprof_ng_module")

xprof_ng_module(
    name = "execution_preview",
    srcs = ["execution_preview.component.ts"],
    assets = [
        "execution_preview.component.css",
        "execution_preview.component.html",
    ],
    visibility = ["//frontend/app/components/kernel_analysis:__pkg__"],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_form_field",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_input",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/components/kernel_analysis/data",
    ],
)
