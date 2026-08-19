load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = [
    "//plugin/xprof:__subpackages__",
])

pytype_strict_library(
    name = "xprof_data",
    srcs = ["xprof_data.py"],
    deps = [
        ":decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/protobuf:op_profile_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "decorators",
    srcs = ["decorators.py"],
    deps = ["@absl_py//absl/logging"],
)

pytype_strict_library(
    name = "numerical_generator",
    srcs = ["numerical_generator.py"],
    visibility = [
        "//experimental/xla/rosetta:__subpackages__",
        "//learning/performance/agents:__subpackages__",
        "//plugin/xprof:__subpackages__",
    ],
    deps = [
        "//third_party/py/ml_dtypes",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "numerical_validator",
    srcs = ["numerical_validator.py"],
    visibility = [
        "//experimental/xla/rosetta:__subpackages__",
        "//learning/performance/agents:__subpackages__",
        "//plugin/xprof:__subpackages__",
    ],
    deps = [
        ":numerical_generator",
        "//third_party/py/ml_dtypes",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "numerical_generator_test",
    srcs = ["numerical_generator_test.py"],
    deps = [
        ":numerical_generator",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "numerical_validator_test",
    srcs = ["numerical_validator_test.py"],
    deps = [
        ":numerical_validator",
        "//third_party/py/ml_dtypes",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
