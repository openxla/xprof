load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = [
    "//plugin/xprof:__subpackages__",
])

pytype_strict_library(
    name = "get_kernel_stats_tool",
    srcs = ["get_kernel_stats_tool.py"],
    visibility = ["//plugin/xprof:__subpackages__"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:kernel_stats_tools",
    ],
)

pytype_strict_library(
    name = "check_host_boundness_tool",
    srcs = ["check_host_boundness_tool.py"],
    visibility = ["//plugin/xprof:__subpackages__"],
    deps = [
        ":get_utilization_viewer_tool",
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_overview_tool",
    srcs = ["get_overview_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_roofline_model_tool",
    srcs = ["get_roofline_model_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_step_trace_tool",
    srcs = ["get_step_trace_tool.py"],
    visibility = ["//plugin/xprof:__subpackages__"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_memory_profile_tool",
    srcs = ["get_memory_profile_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_llo_analysis_tool",
    srcs = ["get_llo_analysis_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

pytype_strict_library(
    name = "get_llo_debug_string_tool",
    srcs = ["get_llo_debug_string_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

pytype_strict_library(
    name = "get_peak_allocations_tool",
    srcs = ["get_peak_allocations_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_kernel_utilization_tool",
    srcs = ["get_kernel_utilization_tool.py"],
    visibility = [
        "//plugin/xprof:__subpackages__",
    ],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/convert:raw_to_tool_data",
    ],
)

pytype_strict_library(
    name = "get_utilization_viewer_tool",
    srcs = ["get_utilization_viewer_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_top_hlo_ops_tool",
    srcs = ["get_top_hlo_ops_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/protobuf:op_profile_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "get_hlo_stats_tool",
    srcs = ["get_hlo_stats_tool.py"],
    visibility = ["//plugin/xprof:__subpackages__"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/protobuf:hlo_stats_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "get_kpi_metrics_tool",
    srcs = ["get_kpi_metrics_tool.py"],
    deps = [
        ":get_memory_profile_tool",
        ":get_overview_tool",
    ],
)

pytype_strict_library(
    name = "get_graph_viewer_tool",
    srcs = ["get_graph_viewer_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "detect_layout_mismatch_copies_tool",
    srcs = ["detect_layout_mismatch_copies_tool.py"],
    deps = [
        ":get_top_hlo_ops_tool",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "detect_unfused_reshapes_tool",
    srcs = ["detect_unfused_reshapes_tool.py"],
    deps = [
        ":get_top_hlo_ops_tool",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "detect_unfused_updates_tool",
    srcs = ["detect_unfused_updates_tool.py"],
    deps = [
        ":get_top_hlo_ops_tool",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "detect_unnecessary_convert_dynamic_scale_tool",
    srcs = ["detect_unnecessary_convert_dynamic_scale_tool.py"],
    deps = [
        ":get_top_hlo_ops_tool",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "detect_unnecessary_convert_reduce_tool",
    srcs = ["detect_unnecessary_convert_reduce_tool.py"],
    deps = [
        ":get_top_hlo_ops_tool",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "verify_numerical_parity_tool",
    srcs = ["verify_numerical_parity_tool.py"],
    deps = [
        "//plugin/xprof/cli/internal:numerical_validator",
    ],
)
