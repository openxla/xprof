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
        "//third_party/py/pandas",
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

pytype_strict_contrib_test(
    name = "get_smart_suggestions_tool_test",
    srcs = ["get_smart_suggestions_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:smart_suggestion_tools",
        "//plugin/xprof/cli/tools:get_smart_suggestions_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "detect_layout_mismatch_copies_tool_test",
    srcs = ["detect_layout_mismatch_copies_tool_test.py"],
    deps = [
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/tools:detect_layout_mismatch_copies_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@xla//xla:xla_data_proto_py",
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
        "//plugin/xprof/cli/internal/oss:xplane_tools",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "detect_unnecessary_convert_reduce_tool_test",
    srcs = ["detect_unnecessary_convert_reduce_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/tools:detect_unnecessary_convert_reduce_tool",
        "@absl_py//absl/testing:absltest",
        "@xla//xla:xla_data_proto_py",
        "@xla//xla/python:xla_client",
        "@xla//xla/service:hlo_proto_py",
    ],
)

pytype_strict_contrib_test(
    name = "detect_unfused_updates_tool_test",
    srcs = ["detect_unfused_updates_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/tools:detect_unfused_updates_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "detect_unfused_reshapes_tool_test",
    srcs = ["detect_unfused_reshapes_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/tools:detect_unfused_reshapes_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "detect_unnecessary_convert_dynamic_scale_tool_test",
    srcs = ["detect_unnecessary_convert_dynamic_scale_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/tools:detect_unnecessary_convert_dynamic_scale_tool",
        "@absl_py//absl/testing:absltest",
        "@xla//xla/python:xla_client",
        "@xla//xla/service:hlo_proto_py",
    ],
)

pytype_strict_contrib_test(
    name = "tolerance_dilemma_test",
    srcs = ["tolerance_dilemma_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:numerical_generator",
        "//plugin/xprof/cli/internal:numerical_validator",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@pypi//jax",
    ],
)

pytype_strict_contrib_test(
    name = "numerical_generator_test",
    srcs = ["numerical_generator_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:numerical_generator",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "verify_numerical_parity_tool_test",
    srcs = ["verify_numerical_parity_tool_test.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/tools:verify_numerical_parity_tool",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
