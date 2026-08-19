load("@rules_python//python:defs.bzl", "py_test")
load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin/xprof:__subpackages__"])

py_test(
    name = "get_kernel_stats_tool_test",
    srcs = ["get_kernel_stats_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:events_db_tools",
        "//plugin/xprof/cli/tools:get_kernel_stats_tool",
        "//plugin/xprof/cli/tools/oss:f1_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "oss_kernel_stats_tools_test",
    srcs = ["oss_kernel_stats_tools_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal/oss:kernel_stats_tools",
        "//plugin/xprof/cli/internal/oss:xplane_tools",
    ],
)

py_test(
    name = "get_memory_profile_tool_test",
    srcs = ["get_memory_profile_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_memory_profile_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "get_peak_allocations_tool_test",
    srcs = ["get_peak_allocations_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_peak_allocations_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "get_top_hlo_ops_tool_test",
    srcs = ["get_top_hlo_ops_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_top_hlo_ops_tool",
        "//plugin/xprof/protobuf:op_profile_proto_py",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "get_kpi_metrics_tool_test",
    srcs = ["get_kpi_metrics_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_kpi_metrics_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "get_overview_tool_test",
    srcs = ["get_overview_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_overview_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "get_graph_viewer_tool_test",
    srcs = ["get_graph_viewer_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_graph_viewer_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "xprof_data_test",
    srcs = ["xprof_data_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal:xprof_data",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/protobuf:op_profile_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "xprof_cli_test",
    srcs = ["xprof_cli_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli:xprof_cli_lib",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "get_utilization_viewer_tool_test",
    srcs = ["get_utilization_viewer_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:get_utilization_viewer_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "oss_xprof_client_test",
    srcs = ["oss_xprof_client_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "oss_xplane_tools_test",
    srcs = ["oss_xplane_tools_test.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xplane_tools",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "oss_hlo_tools_test",
    srcs = ["oss_hlo_tools_test.py"],
    deps = [
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "@absl_py//absl/testing:absltest",
    ],
)
