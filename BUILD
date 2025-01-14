load("@python//:defs.bzl", "compile_pip_requirements")
load("@python_deps//:requirements.bzl", "requirement")
load("@repository_configuration//:repository_config.bzl", "PROFILER_REQUIREMENTS_FILE")

# Description
# Xprof, ML Performance Toolbox (for TPU, GPU, CPU).

licenses(["notice"])

exports_files(["LICENSE"])  # Needed for internal repo.

exports_files([
    "tsconfig.json",
    "rollup.config.js",
])

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
