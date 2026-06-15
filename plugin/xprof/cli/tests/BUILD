load("@rules_python//python:defs.bzl", "py_test")
load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin:__subpackages__"])

py_test(
    name = "get_memory_profile_tool_test",
    srcs = ["get_memory_profile_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_memory_profile_tool",
    ],
)

py_test(
    name = "get_peak_allocations_tool_test",
    srcs = ["get_peak_allocations_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_peak_allocations_tool",
    ],
)

py_test(
    name = "get_top_hlo_ops_tool_test",
    srcs = ["get_top_hlo_ops_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_top_hlo_ops_tool",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_py",
    ],
)

py_test(
    name = "get_kpi_metrics_tool_test",
    srcs = ["get_kpi_metrics_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_kpi_metrics_tool",
    ],
)

py_test(
    name = "get_overview_tool_test",
    srcs = ["get_overview_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_overview_tool",
    ],
)

py_test(
    name = "get_graph_viewer_tool_test",
    srcs = ["get_graph_viewer_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_graph_viewer_tool",
    ],
)

pytype_strict_contrib_test(
    name = "xprof_data_test",
    srcs = ["xprof_data_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal:xprof_data",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_py",
    ],
)

pytype_strict_contrib_test(
    name = "xprof_cli_test",
    srcs = ["xprof_cli_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/cli:xprof_cli_lib",
    ],
)

pytype_strict_contrib_test(
    name = "get_smart_suggestions_tool_test",
    srcs = ["get_smart_suggestions_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:smart_suggestion_tools",
        "@org_xprof//plugin/xprof/cli/tools:get_smart_suggestions_tool",
    ],
)

py_test(
    name = "get_utilization_viewer_tool_test",
    srcs = ["get_utilization_viewer_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:get_utilization_viewer_tool",
    ],
)

py_test(
    name = "oss_xprof_client_test",
    srcs = ["oss_xprof_client_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)
