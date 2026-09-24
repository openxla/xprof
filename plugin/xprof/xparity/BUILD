load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = [
    "//plugin/xprof:__subpackages__",
    "@org_xprof//plugin:__subpackages__",
])

licenses(["notice"])

pytype_strict_library(
    name = "numerical_generator",
    srcs = ["numerical_generator.py"],
)

pytype_strict_library(
    name = "numerical_validator",
    srcs = ["numerical_validator.py"],
    deps = [
        ":numerical_generator",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "xparity_tool",
    srcs = ["xparity_tool.py"],
    deps = [
        ":numerical_generator",
        ":numerical_validator",
    ],
)

pytype_strict_library(
    name = "xparity",
    srcs = ["__init__.py"],
    deps = [
        ":numerical_generator",
        ":numerical_validator",
        ":xparity_tool",
    ],
)

pytype_strict_library(
    name = "xparity_cli_lib",
    srcs = ["xparity_cli.py"],
    deps = [
        ":xparity_tool",
        "@absl_py//absl/flags",
        "@pypi//absl_py",
        "@pypi//fire",
    ],
)

pytype_strict_binary(
    name = "xparity_cli",
    srcs = ["xparity_cli.py"],
    main = "xparity_cli.py",
    deps = [
        ":xparity_cli_lib",
    ],
)
