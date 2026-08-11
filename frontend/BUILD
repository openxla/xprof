load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "npm_sass_library")
load("//defs:defs.bzl", "xprof_ng_module")

package(default_visibility = [
    "//frontend:__subpackages__",
    "//plugin:__subpackages__",
])

licenses(["notice"])

package_group(
    name = "internal",
    packages = [
        "//frontend/...",
    ],
)

xprof_ng_module(
    name = "main",
    srcs = [
        "main.ts",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/localize",
        "@npm//@angular/platform-browser",
        "@npm//@angular/platform-browser-dynamic",
        "@org_xprof//frontend/app",
    ],
)

rollup_bundle(
    name = "zone",
    srcs = ["@npm//:node_modules/zone.js/bundles/zone.umd.js"],
    config_file = "//:rollup.config.js",
    entry_point = "@npm//:node_modules/zone.js/bundles/zone.umd.js",
    format = "iife",
    link_workspace_root = True,
    deps = [
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
    ],
)

rollup_bundle(
    name = "bundle",
    config_file = "//:rollup.config.js",
    entry_point = "main.ts",
    format = "iife",
    link_workspace_root = True,
    deps = [
        ":main",
        "@npm//@angular/animations",
        "@npm//@angular/cdk",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@npm//@angular/material",
        "@npm//@angular/platform-browser-dynamic",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
    ],
)

npm_sass_library(
  name = "angular_material_sass_deps",
  deps = ["@npm//@angular/material"],
)

sass_binary(
    name = "css",
    src = "styles.scss",
    # @use was very different in angular v13.x using a relative path @use '../../../cdk/a11y'
    # instead to the new one in v14.x @use '@angular/cdk'
    # need to include the node_modules path to enable sass to find the theming file
    include_paths = [
      "node_modules",
      # used for resolve the external @npm dependencies
      "external/npm/node_modules",
    ],
    output_name = "styles.css",
    sourcemap = False,
    deps = [
        ":angular_material_sass_deps",
    ],
)

py_binary(
    name = "server",
    testonly = True,
    srcs = ["server.py"],
    data = [
        "index.html",
        ":bundle.js",
        ":css",
        ":zone.js",
    ],
    python_version = "PY3",
)
