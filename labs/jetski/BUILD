load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "xprof_py_runner_lib",
    srcs = [
        "__init__.py",
        "xprof_py_runner.py",
    ],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "xprof_py_runner",
    srcs = ["xprof_py_runner.py"],
    deps = [
        ":xprof_py_runner_lib",
    ],
)

py_test(
    name = "xprof_py_runner_test",
    size = "small",
    srcs = ["xprof_py_runner_test.py"],
    deps = [
        ":xprof_py_runner_lib",
        "//third_party/py/matplotlib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:flagsaver",
    ],
)
