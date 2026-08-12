load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "source_code_editor",
    srcs = [
        "source_code_editor.ts",
        "source_code_editor_module.ts",
    ],
    assets = [
        ":source_code_editor_css",
        "source_code_editor.ng.html",
    ],
    deps = [
        "//third_party/javascript/monaco_editor/v0_52_0:gstatic_loader",
        "//third_party/javascript/monaco_editor/v0_52_0:typings",
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_spinner",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/services/source_code_service:source_code_service_interface",
    ],
)

sass_binary(
    name = "source_code_editor_css",
    src = "source_code_editor.scss",
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

xprof_ng_module(
    name = "tests",
    testonly = True,
    srcs = [
        "source_code_editor_test.ts",
    ],
    deps = [
        ":source_code_editor",
        "//third_party/javascript/typings/jasmine",
        "@npm//@angular/core_testing",
    ],
)
