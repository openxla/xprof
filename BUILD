load("@aspect_bazel_lib//lib:copy_to_bin.bzl", "copy_to_bin")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@npm//:defs.bzl", "npm_link_all_packages")
load("@repository_configuration//:repository_config.bzl", "PROFILER_REQUIREMENTS_FILE")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

npm_link_all_packages(name = "node_modules")

# Description
# XProf, ML Performance Toolbox (for TPU, GPU, CPU).

licenses(["notice"])

exports_files(["LICENSE"])  # Needed for internal repo.

exports_files([
    "README.md",
    "requirements.in",
])  # Needed for pip package description and requirements

exports_files([
    "pnpm-workspace.yaml",
    "rollup.config.js",
    "tsconfig.json",
])

bool_flag(
    name = "enable_embedded_features",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "embedded_features_enabled",
    flag_values = {":enable_embedded_features": "True"},
    visibility = ["//visibility:public"],
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

copy_to_bin(
    name = "tsconfig",
    srcs = ["tsconfig.json"],
    visibility = ["//frontend:__subpackages__"],
)

copy_to_bin(
    name = "package_json",
    srcs = ["package.json"],
    visibility = ["//frontend:__subpackages__"],
)
