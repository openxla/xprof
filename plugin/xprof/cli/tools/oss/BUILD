load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["@org_xprof//plugin:__subpackages__"])

pytype_strict_library(
    name = "get_graph_viewer_tool",
    srcs = [
        "__init__.py",
        "get_graph_viewer_tool.py",
    ],
    deps = [
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/internal/oss:xprof_client",
    ],
)

pytype_strict_library(
    name = "get_kernel_utilization_tool",
    srcs = [
        "__init__.py",
        "get_kernel_utilization_tool.py",
    ],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:kernel_stats_tools",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/convert:raw_to_tool_data",
    ],
)

pytype_strict_library(
    name = "upload_trace_tool",
    srcs = [
        "__init__.py",
        "upload_trace_tool.py",
    ],
    deps = ["//plugin/xprof/cli/internal/oss:xprof_client"],
)

pytype_strict_library(
    name = "events_db_tool",
    srcs = ["events_db_tool.py"],
    deps = [
        "//plugin/xprof:version",
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//xprof/convert/events_db/python:pywrap_events_db_c_api_py",
    ],
)
