# load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_visibility = ["//visibility:private"],
    features = ["-use_header_modules"],
    licenses = ["notice"],
)

cc_library(
    name = "pywrap_events_db_c_api",
    srcs = ["pywrap_events_db_c_api.cc"],
    hdrs = ["pywrap_events_db_c_api.h"],
    copts = ["-fvisibility=hidden"],
    linkstatic = 1,
    deps = [
        "@arrow//:arrow_core",
        "@arrow//:util_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/convert/events_db:parquet_record_consumer",
        "@org_xprof//xprof/convert/events_db:record_consumer",
        "@org_xprof//xprof/convert/events_db:schema",
        "@org_xprof//xprof/convert/events_db:xspace_parser",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libpywrap_events_db_c_api.so",
    linkshared = True,
    deps = [":pywrap_events_db_c_api"],
)

cc_test(
    name = "pywrap_events_db_c_api_test",
    srcs = ["pywrap_events_db_c_api_test.cc"],
    deps = [
        ":pywrap_events_db_c_api",
        "@arrow//:util_hdrs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//xprof/convert/events_db:parquet_record_consumer",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

pytype_strict_library(
    name = "pywrap_events_db_c_api_py",
    srcs = ["pywrap_events_db_c_api.py"],
    data = [":libpywrap_events_db_c_api.so"],
    visibility = ["//visibility:public"],
    deps = [
        "@pypi//etils",
        "@rules_python//python/runfiles",
    ],
)

pytype_strict_contrib_test(
    name = "pywrap_events_db_c_api_test_py",
    srcs = ["pywrap_events_db_c_api_test.py"],
    data = ["test_data/test.xplane.pb"],
    main = "pywrap_events_db_c_api_test.py",
    deps = [
        ":pywrap_events_db_c_api_py",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@rules_python//python/runfiles",
    ],
)
