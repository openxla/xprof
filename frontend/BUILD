load("@aspect_bazel_lib//lib:copy_file.bzl", "copy_file")
load("@aspect_bazel_lib//lib:directory_path.bzl", "directory_path")
load("@aspect_rules_rollup//rollup:defs.bzl", "rollup")
load("//defs:defs.bzl", "xprof_ng_module")
load("//defs:sass.bzl", "sass_binary")

package(default_visibility = [
    "//frontend:__subpackages__",
    "//plugin:__subpackages__",
])

licenses(["notice"])

exports_files([
    "rollup.config.js",
])

package_group(
    name = "internal",
    packages = [
        "//frontend/...",
    ],
)

xprof_ng_module(
    name = "main",
    srcs = [
        "main.js",
    ],
    deps = [
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/localize",
        "//:node_modules/@angular/platform-browser-dynamic",
        "//frontend/app",
    ],
)

directory_path(
    name = "zone_dir",
    directory = "//:node_modules/zone.js/dir",
    path = "dist/zone.js",
)

copy_file(
    name = "zone_file",
    src = ":zone_dir",
    out = "zone_out.js",
)

rollup(
    name = "zone",
    config_file = "//frontend:rollup.config.js",
    data = [":zone_file"],
    entry_point = "zone_out.js",
    format = "iife",
    node_modules = "//:node_modules",
    tags = [
        "local",
    ],
    deps = [
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
    ],
)

rollup(
    name = "bundle",
    config_file = "//frontend:rollup.config.js",
    entry_point = "main.js",
    format = "iife",
    node_modules = "//:node_modules",
    tags = [
        "local",
    ],
    deps = [
        ":main",
        "//:node_modules/@angular/animations",
        "//:node_modules/@angular/cdk",
        "//:node_modules/@angular/core",
        "//:node_modules/@angular/forms",
        "//:node_modules/@angular/material",
        "//:node_modules/@angular/platform-browser-dynamic",
        "//:node_modules/@angular/router",
        "//:node_modules/@ngrx/store",
        "//:node_modules/@rollup/plugin-commonjs",
        "//:node_modules/@rollup/plugin-node-resolve",
    ],
)

sass_binary(
    name = "css",
    srcs = ["styles.scss"],
    # output_name = "styles.css",
    # sourcemap = False,
    deps = [
        "//:node_modules/@angular/material",
    ],
)

py_binary(
    name = "server",
    testonly = True,
    srcs = ["server.py"],
    data = [
        "index.html",
        ":bundle",
        ":css",
        ":zone",
    ],
    python_version = "PY3",
)
