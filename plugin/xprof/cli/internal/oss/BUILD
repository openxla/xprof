load("@rules_proto//proto:defs.bzl", "proto_descriptor_set")
load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin/xprof/cli:__subpackages__"])

# Schema of xla.HloProto, loaded into a private DescriptorPool by hlo_tools.
proto_descriptor_set(
    name = "hlo_descriptor_set",
    visibility = ["//plugin:internal"],
    deps = ["@xla//xla/service:hlo_proto"],
)

pytype_library(
    name = "hlo_tools",
    srcs = ["hlo_tools.py"],
    data = [":hlo_descriptor_set"],
    # strict_deps = False,
    deps = [
        ":xprof_client",  # buildcleaner: keep
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal:hlo_graph_db",
        "//plugin/xprof/convert:raw_to_tool_data",
        "@pypi//protobuf",
    ],
)

pytype_library(
    name = "xplane_tools",
    srcs = ["xplane_tools.py"],
    # strict_deps = False,
    deps = [
        ":xprof_client",  # buildcleaner: keep
        "//plugin/xprof:profile_data",  # buildcleaner: keep
        "//plugin/xprof/cli/internal:async_overlap",
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal:xplane_event_query",
    ],
)

pytype_library(
    name = "kernel_stats_tools",
    srcs = ["kernel_stats_tools.py"],
    # strict_deps = False,
    deps = [
        ":xplane_tools",
        "//plugin/xprof/cli/internal:decorators",
    ],
)

pytype_library(
    name = "xprof_client",
    srcs = ["xprof_client.py"],
    # strict_deps = False,
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/xprof/cli/internal:decorators",
        "//plugin/xprof/cli/internal:trace_selection",
        "//plugin/xprof/convert:raw_to_tool_data",
    ],
)

pytype_library(
    name = "smart_suggestion_tools",
    srcs = ["smart_suggestion_tools.py"],
    # strict_deps = False,
    deps = [":xprof_client"],  # buildcleaner: keep
)

pytype_library(
    name = "llo_parser",
    srcs = ["llo_parser.py"],
)

pytype_library(
    name = "llo_detectors",
    srcs = ["llo_detectors.py"],
)

pytype_library(
    name = "llo_report_generator",
    srcs = ["llo_report_generator.py"],
    # strict_deps = False,
    deps = [":llo_detectors"],
)

pytype_library(
    name = "events_db_sql",
    srcs = ["events_db_sql.py"],
    deps = [
        "//plugin/xprof/cli/tools/oss:events_db_tool",
        "@pypi//duckdb",
    ],
)
