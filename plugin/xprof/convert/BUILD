load("@python_deps//:requirements.bzl", "requirement")

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
    deps = [
        ":raw_to_tool_data",
        ":trace_events_json",
    ],
)

py_library(
    name = "trace_events_json",
    srcs = ["trace_events_json.py"],
    deps = [
        "@org_pythonhosted_six",
    ],
)

py_test(
    name = "trace_events_json_test",
    size = "small",
    srcs = ["trace_events_json_test.py"],
    main = "trace_events_json_test.py",
    deps = [
        ":trace_events_json",
        requirement("absl-py"),  # build_cleaner: keep; go/disable_tf2
        "//:expect_tensorflow_installed",
        "//net/proto2/python/public",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "@org_xprof//plugin/xprof/protobuf:trace_events_old_proto_py",
    ],
)

py_library(
    name = "tf_data_stats_proto_to_gviz",
    srcs = ["tf_data_stats_proto_to_gviz.py"],
    deps = [
        requirement("gviz_api"),
        "@org_xprof//plugin/xprof/protobuf:tf_data_stats_proto_py",
    ],
)

py_test(
    name = "tf_data_stats_proto_to_gviz_test",
    size = "small",
    srcs = ["tf_data_stats_proto_to_gviz_test.py"],
    main = "tf_data_stats_proto_to_gviz_test.py",
    deps = [
        ":tf_data_stats_proto_to_gviz",
        requirement("absl-py"),  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        requirement("gviz_api"),
        "//:expect_tensorflow_installed",
        "@org_xprof//plugin/xprof/protobuf:tf_data_stats_proto_py",
    ],
)

py_library(
    name = "raw_to_tool_data",
    srcs = ["raw_to_tool_data.py"],
    visibility = visibility + ["//perftools/accelerators/xprof/benchmark/xprof_benchmarking_tools:__subpackages__"],
    deps = [
        ":csv_writer",
        ":trace_events_json",
        "@org_xprof//plugin/xprof/protobuf:trace_events_old_proto_py",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

py_test(
    name = "raw_to_tool_data_test",
    srcs = ["raw_to_tool_data_test.py"],
    deps = [
        ":raw_to_tool_data",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "csv_writer",
    srcs = ["csv_writer.py"],
)

py_test(
    name = "csv_writer_test",
    srcs = ["csv_writer_test.py"],
    deps = [":csv_writer"],
)
