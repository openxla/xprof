load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//xprof:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "xspace_to_parquet_lib",
    srcs = ["xspace_to_parquet.py"],
    deps = [
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

pytype_strict_binary(
    name = "xspace_to_parquet",
    srcs = ["xspace_to_parquet.py"],
    deps = [
        ":xspace_to_parquet_lib",
    ],
)

pytype_strict_contrib_test(
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

pytype_strict_library(
    name = "count_zero_self_time_events_lib",
    srcs = ["count_zero_self_time_events.py"],
    deps = [
        "//third_party/py/absl:app",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

pytype_strict_binary(
    name = "count_zero_self_time_events",
    srcs = ["count_zero_self_time_events.py"],
    deps = [
        ":count_zero_self_time_events_lib",
    ],
)

pytype_strict_contrib_test(
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

pytype_strict_library(
    name = "zero_self_time_ratio_lib",
    srcs = ["zero_self_time_ratio.py"],
    deps = [
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@org_xprof//xprof/convert/events_db/python:events_db",
        "@pypi//etils",
    ],
)

pytype_strict_binary(
    name = "zero_self_time_ratio",
    srcs = ["zero_self_time_ratio.py"],
    deps = [
        ":zero_self_time_ratio_lib",
    ],
)

pytype_strict_contrib_test(
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
