# Description:
#  Tools for building the Profile Tensorboard plugin pip package.
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
        "setup.py",
        "//:README.md",
        "@org_xprof//frontend:bundle.js",
        "@org_xprof//frontend:css",
        "@org_xprof//frontend:zone.js",
        "@org_xprof//plugin/xprof:resources",
        "@org_xprof//plugin/xprof:server",
        "@org_xprof//plugin/xprof:version",
        "@org_xprof//plugin/xprof/convert:all_libs",
        "@org_xprof//plugin/xprof/integration_tests/tpu/tensorflow:tpu_tf2_keras_xla",
        "@org_xprof//plugin/xprof/protobuf:protos_all",
        "@org_xprof//plugin/trace_viewer:trace_viewer_index.html",
        "@org_xprof//plugin/trace_viewer:trace_viewer_index.js",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
    ],
    tags = [
        "local",
        "manual",
    ],
)
