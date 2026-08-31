load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = [
    "//gdm/sys/l2p/exploratory/lumini:__subpackages__",
    "//plugin:internal",
    "@org_xprof//plugin:__subpackages__",
])

pytype_strict_library(
    name = "xprof_cli_lib",
    srcs = ["xprof_cli.py"],
    deps = [
        "//plugin/xprof:server",
        "//plugin/xprof/cli/internal:xprof_data",
        "//plugin/xprof/cli/internal/oss:hlo_tools",
        "//plugin/xprof/cli/internal/oss:xplane_tools",
        "//plugin/xprof/cli/internal/oss:xprof_client",
        "//plugin/xprof/cli/tools:check_host_boundness_tool",
        "//plugin/xprof/cli/tools:get_graph_viewer_tool",
        "//plugin/xprof/cli/tools:get_hlo_stats_tool",
        "//plugin/xprof/cli/tools:get_kernel_stats_tool",
        "//plugin/xprof/cli/tools:get_kernel_utilization_tool",
        "//plugin/xprof/cli/tools:get_kpi_metrics_tool",
        "//plugin/xprof/cli/tools:get_llo_analysis_tool",
        "//plugin/xprof/cli/tools:get_llo_debug_string_tool",
        "//plugin/xprof/cli/tools:get_memory_profile_tool",
        "//plugin/xprof/cli/tools:get_overview_tool",
        "//plugin/xprof/cli/tools:get_peak_allocations_tool",
        "//plugin/xprof/cli/tools:get_roofline_model_tool",
        "//plugin/xprof/cli/tools:get_top_hlo_ops_tool",
        "//plugin/xprof/cli/tools:get_utilization_viewer_tool",
        "//plugin/xprof/cli/tools:verify_numerical_parity_tool",
        "@absl_py//absl/flags",
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
