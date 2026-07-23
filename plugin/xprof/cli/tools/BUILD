load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

pytype_strict_library(
    name = "get_overview_tool",
    srcs = ["get_overview_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_memory_profile_tool",
    srcs = ["get_memory_profile_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_llo_analysis_tool",
    srcs = ["get_llo_analysis_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

pytype_strict_library(
    name = "get_llo_debug_string_tool",
    srcs = ["get_llo_debug_string_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
)

pytype_strict_library(
    name = "get_peak_allocations_tool",
    srcs = ["get_peak_allocations_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_utilization_viewer_tool",
    srcs = ["get_utilization_viewer_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_top_hlo_ops_tool",
    srcs = ["get_top_hlo_ops_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "get_kpi_metrics_tool",
    srcs = ["get_kpi_metrics_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        ":get_memory_profile_tool",
        ":get_overview_tool",
    ],
)

pytype_strict_library(
    name = "get_graph_viewer_tool",
    srcs = ["get_graph_viewer_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = ["@org_xprof//plugin/xprof/cli/internal/oss:xprof_client"],
)

pytype_strict_library(
    name = "get_smart_suggestions_tool",
    srcs = ["get_smart_suggestions_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/cli/internal/oss:smart_suggestion_tools",
    ],
)

pytype_strict_library(
    name = "detect_unfused_reshapes_tool",
    srcs = ["detect_unfused_reshapes_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        ":get_top_hlo_ops_tool",
        "@org_xprof//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "detect_layout_mismatch_copies_tool",
    srcs = ["detect_layout_mismatch_copies_tool.py"],
    visibility = [
        "//plugin:__subpackages__",
    ],
    deps = [
        ":get_top_hlo_ops_tool",
        "@org_xprof//plugin/xprof/cli/internal/oss:hlo_tools",
        "@xla//xla/python:xla_client",
    ],
)

pytype_strict_library(
    name = "detect_unnecessary_convert_reduce_tool",
    srcs = ["detect_unnecessary_convert_reduce_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        ":get_top_hlo_ops_tool",
        "@org_xprof//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)

pytype_strict_library(
    name = "detect_unfused_updates_tool",
    srcs = ["detect_unfused_updates_tool.py"],
    visibility = ["//plugin:__subpackages__"],
    deps = [
        ":get_top_hlo_ops_tool",
        "@org_xprof//plugin/xprof/cli/internal/oss:hlo_tools",
    ],
)
