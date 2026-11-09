load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

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
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/classes",
        "//frontend/app/common/constants",
        "//frontend/app/common/utils",
        "//frontend/app/pipes",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "megascale_perfetto_css",
    src = "megascale_perfetto.scss",
)
