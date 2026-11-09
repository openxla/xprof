load("//defs:defs.bzl", "sass_binary", "sass_library", "ts_library", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "memory_profile",
    srcs = [
        "memory_profile.ts",
        "memory_profile_module.ts",
    ],
    assets = [
        ":memory_profile_css",
        "memory_profile.ng.html",
    ],
    deps = [
        ":memory_profile_base",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/rxjs",
        "//frontend/app/common/angular:angular_material_core",
        "//frontend/app/common/angular:angular_material_divider",
        "//frontend/app/common/angular:angular_material_icon",
        "//frontend/app/common/angular:angular_material_progress_bar",
        "//frontend/app/common/angular:angular_material_select",
        "//frontend/app/common/classes",
        "//frontend/app/common/interfaces",
        "//frontend/app/components/memory_profile/memory_breakdown_table",
        "//frontend/app/components/memory_profile/memory_profile_summary",
        "//frontend/app/components/memory_profile/memory_timeline_graph",
        "//frontend/app/services/data_service_v2:data_service_v2_interface",
        "//frontend/app/store",
    ],
)

ts_library(
    name = "memory_profile_base",
    srcs = [
        "memory_profile_base.ts",
    ],
    deps = [
        "//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "memory_profile_css",
    src = "memory_profile.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        ":memory_profile_common",
    ],
)

sass_library(
    name = "memory_profile_common",
    srcs = ["memory_profile_common.scss"],
    deps = [
        "//frontend/app/styles:common",
    ],
)
