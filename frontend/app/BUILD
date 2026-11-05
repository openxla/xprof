load("//defs:defs.bzl", "sass_binary", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "app",
    srcs = [
        "app.ts",
        "app_module.ts",
    ],
    assets = [
        ":app_css",
        "app.ng.html",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/platform-browser",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_common_http",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_platform-browser_animation",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/empty_page",
        "//frontend/app/components/main_page",
        "//frontend/app/pipes",
        "//frontend/app/services/data_dispatcher",
        "//frontend/app/services/data_service_v2",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/services/source_code_service",
        "//frontend/app/services/source_code_service:source_code_service_interface",
        "//frontend/app/store",
    ],
)

sass_binary(
    name = "app_css",
    src = "app.scss",
    # stack = False,
    sourcemap = False,
)
