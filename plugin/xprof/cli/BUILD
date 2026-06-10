load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin:__subpackages__"])

pytype_strict_library(
    name = "xprof_cli_lib",
    srcs = ["xprof_cli.py"],
    deps = [
        "@absl_py//absl/flags",
        "@org_xprof//plugin/xprof:server",
        "@org_xprof//plugin/xprof/cli/internal:xprof_data",
        "@org_xprof//plugin/xprof/cli/internal/oss:hlo_tools",
        "@org_xprof//plugin/xprof/cli/internal/oss:xplane_tools",
        "@org_xprof//plugin/xprof/cli/internal/oss:xprof_client",
        "@org_xprof//plugin/xprof/cli/tools:detect_unfused_reshapes_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_graph_viewer_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_kpi_metrics_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_memory_profile_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_overview_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_peak_allocations_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_smart_suggestions_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_top_hlo_ops_tool",
        "@org_xprof//plugin/xprof/cli/tools:get_utilization_viewer_tool",
        "@org_xprof//plugin/xprof/cli/tools/oss:upload_trace_tool",
        "@pypi//absl_py",
        "@pypi//fire",
    ],
)

pytype_strict_binary(
    name = "xprof_cli",
    srcs = ["xprof_cli.py"],
    deps = [
        ":xprof_cli_lib",
        "@absl_py//absl/flags",
        "@pypi//absl_py",
    ],
)
