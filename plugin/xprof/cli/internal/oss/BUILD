load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["@org_xprof//plugin/xprof/cli:__subpackages__"])

pytype_library(
    name = "hlo_tools",
    srcs = ["hlo_tools.py"],
    # strict_deps = False,
    deps = [
        ":xprof_client",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
        "@org_xprof//plugin/xprof/convert:raw_to_tool_data",
    ],
)

pytype_library(
    name = "xplane_tools",
    srcs = ["xplane_tools.py"],
    # strict_deps = False,
    deps = [
        ":xprof_client",
        "@org_xprof//plugin/xprof:profile_data",
        "@org_xprof//plugin/xprof/cli/internal:decorators",
    ],
)

pytype_library(
    name = "xprof_client",
    srcs = ["xprof_client.py"],
    # strict_deps = False,
    visibility = ["//visibility:public"],
    deps = [
        "@org_xprof//plugin/xprof/convert:raw_to_tool_data",
    ],
)

pytype_library(
    name = "smart_suggestion_tools",
    srcs = ["smart_suggestion_tools.py"],
    # strict_deps = False,
    deps = [
        ":xprof_client",
    ],
)
