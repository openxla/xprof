load("@local_xla//xla/tsl:tsl.bzl", pytype_extension = "tsl_pybind_extension_opensource")
load("//plugin/tensorboard_plugin_profile/build_utils:strict.default.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

pytype_extension(
    name = "_pywrap_profiler_plugin",
    srcs = ["pywrap_profiler_plugin.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    pytype_srcs = [
        "_pywrap_profiler_plugin.pyi",
    ],
    deps = [
        ":profiler_plugin_impl",
        "@local_xla//xla/pjrt:status_casters",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/platform/cloud:gcs_file_system",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@org_xprof//xprof/convert:tool_options",
        "@pybind11",
    ],
)

py_strict_test(
    name = "profiler_wrapper_test",
    srcs = ["profiler_wrapper_test.py"],
    tags = [
        "no_pip",
    ],
    deps = [
        ":_pywrap_profiler_plugin",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

cc_library(
    name = "profiler_plugin_impl",
    srcs = ["profiler_plugin_impl.cc"],
    hdrs =
        ["profiler_plugin_impl.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/profiler/utils:session_manager",
        "@org_xprof//xprof/convert:repository",
        "@org_xprof//xprof/convert:tool_options",
        "@org_xprof//xprof/convert:xplane_to_tools_data",
    ],
)
