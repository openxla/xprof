# Description:
#  Tools for building the Profile Tensorboard plugin pip package.
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

package_group(
    name = "internal",
    packages = [
        "//...",
    ],
)

# rm -rf /tmp/profile-pip
# bazel run plugin:build_pip_package
sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "setup.py",
        "//:README.md",
        "//:requirements.in",
        "//:skills_files",
        "//frontend:bundle.js",
        "//frontend:css",
        "//frontend:runtime.js",
        "//frontend:zone.js",
        "//frontend/app/components/trace_viewer_v2:trace_viewer_v2_wasm",
        "//plugin/xprof:profile_data",
        "//plugin/xprof:profile_io",
        "//plugin/xprof:resources",
        "//plugin/xprof:server",
        "//plugin/xprof:version",
        "//plugin/xprof/api:continuous_profiling_snapshot",
        "//plugin/xprof/cli:xprof_cli_lib",
        "//plugin/xprof/convert:all_libs",
        "//plugin/xprof/integration_tests/tpu/tensorflow:tpu_tf2_keras_xla",
        "//plugin/xprof/protobuf:protos_all",
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/trace_viewer:trace_viewer_index.html",
        "//plugin/trace_viewer:trace_viewer_index.js",
        "//xprof/convert/events_db/python:pywrap_events_db_c_api_py",
        "//xprof/pywrap:_pywrap_profiler_plugin",
    ],
    tags = [
        "local",
        "manual",
    ],
)
