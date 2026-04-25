load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@repository_configuration//:repository_config.bzl", "PROFILER_REQUIREMENTS_FILE")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

# Description
# XProf, ML Performance Toolbox (for TPU, GPU, CPU).

licenses(["notice"])

exports_files(["LICENSE"])  # Needed for internal repo.

exports_files(["README.md"])  # Needed for pip package description

exports_files([
    "tsconfig.json",
    "rollup.config.js",
])

bool_flag(
    name = "enable_module_summary",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "module_summary_enabled",
    flag_values = {":enable_module_summary": "True"},
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
