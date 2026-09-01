load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//third_party/nanobind_abseil/oss:nanobind.bzl", "nanobind_extension")

package(
    default_visibility = ["//visibility:private"],
    features = ["-use_header_modules"],
    licenses = ["notice"],
)

nanobind_extension(
    name = "pywrap_events_db",
    srcs = ["pywrap_events_db.cc"],
    deps = [
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/convert/events_db:schema",
    ],
)

py_library(
    name = "events_db",
    srcs = [
        "__init__.py",
        "events_db.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_events_db",
    ],
)

py_test(
    name = "events_db_test",
    srcs = ["events_db_test.py"],
    deps = [
        ":events_db",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
