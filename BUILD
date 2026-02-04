load("@aspect_rules_js//npm:defs.bzl", "npm_link_package", "npm_package")
load("@aspect_rules_ts//ts:defs.bzl", "ts_config")
load("@npm//:defs.bzl", "npm_link_all_packages")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@repository_configuration//:repository_config.bzl", "PROFILER_REQUIREMENTS_FILE")

npm_link_all_packages(
    name = "node_modules",
)

npm_package(
    name = "workspace_pkg",
    srcs = glob(
        ["**/*"],
        exclude = [
            "bazel-*/**",
            "node_modules/**",
            "**/node_modules/**",
        ],
    ),
    package = "org_xprof",
    visibility = ["//visibility:public"],
)

npm_link_package(
    name = "org_xprof",
    src = ":workspace_pkg",
    visibility = ["//visibility:public"],
)

# Description
# XProf, ML Performance Toolbox (for TPU, GPU, CPU).

licenses(["notice"])

exports_files(["LICENSE"])  # Needed for internal repo.

exports_files(["README.md"])  # Needed for pip package description

exports_files([
    "tsconfig.json",
    "rollup.config.js",
])

ts_config(
    name = "tsconfig",
    src = "tsconfig.json",
    visibility = [":__subpackages__"],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a tensorflow dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow`
    visibility = ["//visibility:public"],
)

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    generate_hashes = True,
    requirements_in = "requirements.in",
    requirements_txt = PROFILER_REQUIREMENTS_FILE,
)

platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)
