load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//xprof:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_library(
    name = "xspace_to_parquet_lib",
    srcs = ["xspace_to_parquet.py"],
    deps = [
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

py_binary(
    name = "xspace_to_parquet",
    srcs = ["xspace_to_parquet.py"],
    deps = [
        ":xspace_to_parquet_lib",
    ],
)

py_test(
    name = "xspace_to_parquet_test",
    srcs = ["xspace_to_parquet_test.py"],
    deps = [
        ":xspace_to_parquet_lib",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:flagsaver",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

py_library(
    name = "count_zero_self_time_events_lib",
    srcs = ["count_zero_self_time_events.py"],
    deps = [
        "//third_party/py/absl:app",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

py_binary(
    name = "count_zero_self_time_events",
    srcs = ["count_zero_self_time_events.py"],
    deps = [
        ":count_zero_self_time_events_lib",
    ],
)

py_test(
    name = "count_zero_self_time_events_test",
    srcs = ["count_zero_self_time_events_test.py"],
    data = ["@org_xprof//xprof/convert/events_db/examples/test_data:test.xplane.pb"],
    deps = [
        ":count_zero_self_time_events_lib",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:flagsaver",
        "@pypi//etils",
        "@rules_python//python/runfiles",
    ],
)

py_library(
    name = "zero_self_time_ratio_lib",
    srcs = ["zero_self_time_ratio.py"],
    deps = [
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

py_binary(
    name = "zero_self_time_ratio",
    srcs = ["zero_self_time_ratio.py"],
    deps = [
        ":zero_self_time_ratio_lib",
    ],
)

py_test(
    name = "zero_self_time_ratio_test",
    srcs = ["zero_self_time_ratio_test.py"],
    data = ["@org_xprof//xprof/convert/events_db/examples/test_data:test.xplane.pb"],
    deps = [
        ":zero_self_time_ratio_lib",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:flagsaver",
        "@pypi//etils",
        "@rules_python//python/runfiles",
    ],
)
