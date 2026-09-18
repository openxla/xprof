load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin/xprof:__subpackages__"])

licenses(["notice"])

pytype_strict_contrib_test(
    name = "numerical_generator_test",
    srcs = ["numerical_generator_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "numerical_validator_test",
    srcs = ["numerical_validator_test.py"],
    shard_count = 4,
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "xparity_cli_test",
    srcs = ["xparity_cli_test.py"],
    deps = [
        "//plugin/xprof/xparity",
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "tolerance_dilemma_test",
    srcs = ["tolerance_dilemma_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
