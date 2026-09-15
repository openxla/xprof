load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "execution_preview",
    srcs = ["execution_preview.ts"],
    assets = [
        "execution_preview.css",
        "execution_preview.ng.html",
    ],
    visibility = ["@org_xprof//frontend/app/components/kernel_analysis:__pkg__"],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/components/kernel_analysis/data",
    ],
)
