load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "capture_profile",
    srcs = [
        "capture_profile.ts",
        "capture_profile_module.ts",
    ],
    assets = [
        ":capture_profile_css",
        "capture_profile.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/common",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_button",
        "//frontend/app/common/angular:angular_material_dialog",
        "//frontend/app/common/angular:angular_material_progress_spinner",
        "//frontend/app/common/angular:angular_material_snack_bar",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/capture_profile/capture_profile_dialog",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "capture_profile_css",
    src = "capture_profile.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
