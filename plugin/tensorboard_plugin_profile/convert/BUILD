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
        ":dcn_collective_stats_proto_to_gviz",
        ":diagnostics",
        ":input_pipeline_proto_to_gviz",
        ":overview_page_proto_to_gviz",
        ":raw_to_tool_data",
        ":trace_events_json",
    ],
)

py_library(
    name = "diagnostics",
    srcs = ["diagnostics.py"],
    deps = [
        requirement("gviz_api"),
    ],
)

py_test(
    name = "diagnostics_test",
    size = "small",
    srcs = ["diagnostics_test.py"],
    deps = [
        ":diagnostics",
        "//:expect_tensorflow_installed",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:diagnostics_proto_py",
    ],
)

py_library(
    name = "input_pipeline_proto_to_gviz",
    srcs = ["input_pipeline_proto_to_gviz.py"],
    deps = [
        ":diagnostics",
        requirement("gviz_api"),
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:input_pipeline_proto_py",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tpu_input_pipeline_proto_py",
    ],
)

py_test(
    name = "input_pipeline_proto_to_gviz_test",
    size = "small",
    srcs = ["input_pipeline_proto_to_gviz_test.py"],
    main = "input_pipeline_proto_to_gviz_test.py",
    deps = [
        ":input_pipeline_proto_to_gviz",
        "//google/protobuf:any_py_pb2",
        requirement("absl-py"),  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        requirement("gviz_api"),
        "//:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:input_pipeline_proto_py",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tpu_input_pipeline_proto_py",
    ],
)

py_library(
    name = "overview_page_proto_to_gviz",
    srcs = ["overview_page_proto_to_gviz.py"],
    deps = [
        ":diagnostics",
        ":input_pipeline_proto_to_gviz",
        requirement("gviz_api"),
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:overview_page_proto_py",
    ],
)

py_test(
    name = "overview_page_proto_to_gviz_test",
    size = "small",
    srcs = ["overview_page_proto_to_gviz_test.py"],
    main = "overview_page_proto_to_gviz_test.py",
    deps = [
        ":overview_page_proto_to_gviz",
        requirement("absl-py"),  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        requirement("gviz_api"),
        "//:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:overview_page_proto_py",
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
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:trace_events_old_proto_py",
    ],
)

py_library(
    name = "tf_data_stats_proto_to_gviz",
    srcs = ["tf_data_stats_proto_to_gviz.py"],
    deps = [
        requirement("gviz_api"),
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_data_stats_proto_py",
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
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tf_data_stats_proto_py",
    ],
)

py_library(
    name = "dcn_collective_stats_proto_to_gviz",
    srcs = ["dcn_collective_stats_proto_to_gviz.py"],
    deps = [
        requirement("gviz_api"),
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:dcn_slack_analysis_proto_py",
    ],
)

py_test(
    name = "dcn_collective_stats_proto_to_gviz_test",
    size = "small",
    srcs = ["dcn_collective_stats_proto_to_gviz_test.py"],
    main = "dcn_collective_stats_proto_to_gviz_test.py",
    deps = [
        ":dcn_collective_stats_proto_to_gviz",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        requirement("gviz_api"),
        "//:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:dcn_slack_analysis_proto_py",
    ],
)

py_library(
    name = "raw_to_tool_data",
    srcs = ["raw_to_tool_data.py"],
    visibility = visibility,
    deps = [
        ":csv_writer",
        ":dcn_collective_stats_proto_to_gviz",
        ":inference_stats_proto_to_gviz",
        ":input_pipeline_proto_to_gviz",
        ":overview_page_proto_to_gviz",
        ":trace_events_json",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:trace_events_old_proto_py",
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
    name = "inference_stats_proto_to_gviz",
    srcs = ["inference_stats_proto_to_gviz.py"],
    deps = [
        requirement("gviz_api"),
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:inference_stats_proto_py",
    ],
)

py_library(
    name = "csv_writer",
    srcs = ["csv_writer.py"],
)

py_test(
    name = "csv_writer_test",
    srcs = ["csv_writer_test.py"],
    deps = [
        ":csv_writer",
        "//:expect_tensorflow_installed",
    ],
)
