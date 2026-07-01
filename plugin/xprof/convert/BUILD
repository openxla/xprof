# Description:
#   Converter from protobuf to gviz/json formats.

load("@rules_python//python:defs.bzl", "py_library", "py_test")

visibility = ["//plugin:internal"]

package(
    default_visibility = visibility,
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "all_libs",
    # strict_deps = False,
    deps = [
        ":raw_to_tool_data",
        ":trace_events_json",
    ],
)

py_library(
    name = "trace_events_json",
    srcs = ["trace_events_json.py"],
    # strict_deps = False,
    deps = [
        "@pypi//six",
    ],
)

py_test(
    name = "trace_events_json_test",
    size = "small",
    srcs = ["trace_events_json_test.py"],
    main = "trace_events_json_test.py",
    # strict_deps = False,
    deps = [
        ":trace_events_json",
        "//:expect_tensorflow_installed",
        "//third_party/py/google/protobuf",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "@org_xprof//plugin/xprof/protobuf:trace_events_old_proto_py",
        "@pypi//absl-py",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_library(
    name = "raw_to_tool_data",
    srcs = [
        "raw_to_tool_data.py",
    ],
    # strict_deps = False,
    visibility = visibility,
    deps = [
        ":csv_writer",
        ":trace_events_json",  # build_cleaner: keep
        "@org_xprof//plugin/xprof/protobuf:trace_events_old_proto_py",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

py_test(
    name = "raw_to_tool_data_test",
    size = "small",
    srcs = ["raw_to_tool_data_test.py"],
    main = "raw_to_tool_data_test.py",
    # strict_deps = False,
    deps = [
        ":raw_to_tool_data",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/protobuf:trace_events_old_proto_py",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

py_library(
    name = "csv_writer",
    srcs = ["csv_writer.py"],
    # strict_deps = False,
)

py_test(
    name = "csv_writer_test",
    srcs = ["csv_writer_test.py"],
    # strict_deps = False,
    deps = [":csv_writer"],
)
