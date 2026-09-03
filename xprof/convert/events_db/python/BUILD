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
        "//third_party/arrow:util_hdrs",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/convert/events_db:parquet_record_consumer",
        "@org_xprof//xprof/convert/events_db:record_consumer",
        "@org_xprof//xprof/convert/events_db:schema",
        "@org_xprof//xprof/convert/events_db:xspace_parser",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
    data = ["test_data/test.xplane.pb"],
    deps = [
        ":events_db",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@rules_python//python/runfiles",
    ],
)
