load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "sass_library")
load("@npm//@bazel/concatjs:index.bzl", "ts_library")
load("//defs:defs.bzl", "xprof_ng_module")

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
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/memory_profile/memory_breakdown_table",
        "@org_xprof//frontend/app/components/memory_profile/memory_profile_summary",
        "@org_xprof//frontend/app/components/memory_profile/memory_timeline_graph",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
    ],
)

ts_library(
    name = "memory_profile_base",
    srcs = [
        "memory_profile_base.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "memory_profile_test.ts",
    ],
    deps = [
        ":memory_profile",
        "//javascript/angular2/testing/catalyst/fake_async",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/router",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/interfaces:memory_profile_proto_defs",
        "@org_xprof//frontend/app/common/utils:testing",
        "@org_xprof//frontend/app/services/data_service_v2",
        "@org_xprof//frontend/app/services/data_service_v2:data_service_v2_interface",
        "@org_xprof//frontend/app/store",
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
        "@org_xprof//frontend/app/styles:common",
    ],
)
