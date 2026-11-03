load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "inference_profile",
    srcs = [
        "inference_profile.ts",
        "inference_profile_module.ts",
    ],
    assets = [
        ":inference_profile_css",
        "inference_profile.ng.html",
    ],
    # strict_templates = False,
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@types/google.visualization",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/angular:angular_material_sidenav",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/utils",
        "//frontend/app/components/chart/table",
        "//frontend/app/components/diagnostics_view",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "inference_profile_css",
    src = "inference_profile.scss",
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
