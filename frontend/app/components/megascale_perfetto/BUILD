load("@gzgz_rules_sass//sass:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "megascale_perfetto",
    srcs = [
        "megascale_perfetto.ts",
        "megascale_perfetto_module.ts",
    ],
    assets = [
        ":megascale_perfetto_css",
        "megascale_perfetto.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_sidenav",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/pipes",
        "@org_xprof//frontend/app/store",
    ],
)

sass_binary(
    name = "megascale_perfetto_css",
    src = "megascale_perfetto.scss",
)
