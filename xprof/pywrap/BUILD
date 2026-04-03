# load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")
load("//plugin/xprof/build_utils:strict.default.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
)

cc_binary(
    name = "profiler_plugin_c_api.so",
    srcs = [
        "profiler_plugin_c_api.cc",
        "profiler_plugin_c_api.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fvisibility=hidden",  # Prevent exposing all symbols
    ] + select({
        ":dbg_build": ["-g"],
        "//conditions:default": [],
    }) + select({
        "//xprof:module_summary_enabled": ["-DENABLE_MODULE_SUMMARY"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@xla//xla/tsl:windows": [
            "/OPT:REF",  # Remove unused functions and data
            "/OPT:ICF",  # Merge identical functions
        ],
        "//conditions:default": [
            "-Wl,--exclude-libs,ALL",  # Prevent exposing symbols from dependencies
            "-Wl,--gc-sections",  # Garbage collect any unused sections
            "-Wl,--icf=all",  # Merge identical functions
        ],
    }),
    linkshared = True,
    deps = [
        ":profiler_plugin_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_xprof//plugin/xprof/worker:stub_factory",
        "@org_xprof//xprof/convert:tool_options",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla//xla/tsl/profiler/rpc/client:capture_profile",
    ] + select({
        "//xprof:module_summary_enabled": [
            "@org_xprof//internal/module_summary:module_summary_c_api",
            "@org_xprof//internal/module_summary:refactored_tool_lib",
        ],
        "//conditions:default": [],
    }),
)

py_library(
    name = "_pywrap_profiler_plugin",
    srcs = ["_pywrap_profiler_plugin.py"],
    data = [":profiler_plugin_c_api.so"],
)

py_strict_test(
    name = "profiler_wrapper_test",
    srcs = ["profiler_wrapper_test.py"],
    tags = ["no_oss"],
    deps = [
        ":_pywrap_profiler_plugin",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "profiler_plugin_impl",
    srcs = ["profiler_plugin_impl.cc"],
    hdrs =
        ["profiler_plugin_impl.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@org_xprof//plugin/xprof/worker:grpc_server",
        "@org_xprof//xprof/convert:repository",
        "@org_xprof//xprof/convert:tool_options",
        "@org_xprof//xprof/convert:xplane_to_tools_data",
        "@org_xprof//xprof/convert:xplane_to_tools_data_with_profile_processor",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@xla//xla/tsl/profiler/utils:session_manager",
    ],
)
