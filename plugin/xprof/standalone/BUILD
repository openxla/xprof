# Description:
#   A collection of classes to bypass Tensorboard dependencies.

load("@rules_python//python:defs.bzl", "py_test")
load("@rules_python//python:py_library.bzl", "py_library")

visibility = ["//plugin:internal"]

package(
    default_visibility = visibility,
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "tensorboard_shim",
    srcs = ["tensorboard_shim.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_libs",
    ],
)

py_library(
    name = "all_libs",
    deps = [
        ":base_plugin",
        ":context",
        ":data_provider",
        ":plugin_asset_util",
        ":plugin_event_multiplexer",
    ],
)

py_library(
    name = "base_plugin",
    srcs = ["base_plugin.py"],
    srcs_version = "PY2AND3",
    deps = [":plugin_event_multiplexer"],
)

py_library(
    name = "plugin_event_multiplexer",
    srcs = ["plugin_event_multiplexer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":plugin_asset_util",
        "@pypi//etils",
    ],
)

py_library(
    name = "data_provider",
    srcs = ["data_provider.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "plugin_asset_util",
    srcs = ["plugin_asset_util.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "context",
    srcs = ["context.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_test(
    name = "plugin_asset_util_test",
    srcs = ["plugin_asset_util_test.py"],
    deps = [
        ":plugin_asset_util",
        "//testing/pybase",
        "//testing/pybase:parameterized",
    ],
)
